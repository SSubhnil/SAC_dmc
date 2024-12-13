#!/usr/bin/env python3
import os

from sympy.integrals.meijerint_doc import category

import wandb
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from utils.env_wrapper import DMCConfounderWrapper
from utils.replay_buffer import ReplayBuffer
from logger.logger import Logger
from utils.utils import set_seed_everywhere, eval_mode
from logger.video_recorder import VideoRecorder
from sac.sac import SACAgent
from tqdm import tqdm
import numpy as np
from datetime import datetime
import uuid

# wandb login 576d985d69bfd39f567224809a6a3dd329326993
@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    def make_env(cfg):
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])
        env = DMCConfounderWrapper(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg.seed,
            confounder_params=cfg.confounders[cfg.env],
            action_masking_params=cfg.confounders[cfg.env].get('action_masking', {})
        )
        return env

    # Setup
    work_dir = os.getcwd()
    print(f"Workspace: {work_dir}")

    # Generate unique experiment name to prevent WandB run merging
    unique_id = uuid.uuid4().hex[:8]  # 8-character unique ID
    cfg.experiment = f"{cfg.experiment}_{unique_id}"

    # Update log_dir with the new experiment name
    cfg.log_dir = f"./logs/{cfg.experiment}"
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    env = make_env(cfg)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]

    # Update cfg with dimensions
    cfg.agent.params.obs_dim = obs_dim
    cfg.agent.params.action_dim = action_dim
    cfg.agent.params.action_range = action_range
    cfg.agent.params.critic_cfg.obs_dim = obs_dim
    cfg.agent.params.critic_cfg.action_dim = action_dim
    cfg.agent.params.actor_cfg.obs_dim = obs_dim
    cfg.agent.params.actor_cfg.action_dim = action_dim

    # Initialize WandB
    wandb_run = None
    if cfg.log_save_wandb:
        try:
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,  # Ensure this is your work team name
                name=cfg.experiment,  # Unique run name
                config=OmegaConf.to_container(cfg, resolve=True),
                group=cfg.wandb.group,
                job_type=cfg.wandb.job_type,
                mode=cfg.wandb.mode,
                sync_tensorboard=cfg.wandb.sync_tensorboard,
                reinit=False  # Prevent multiple runs in the same process
            )
            print("WandB initialized successfully.")
        except wandb.errors.CommError as e:
            print(f"WandB Initialization Error: {e}")
            wandb_run = None

    # Initialize Logger
    logger = Logger(
        log_dir=cfg.log_dir,
        save_tb=cfg.log_save_tb,
        log_frequency=cfg.log_frequency,
        agent=cfg.agent.name,
        wandb_run=wandb_run
    )

    # Initialize Agent and Replay Buffer
    agent = SACAgent(**cfg.agent.params)

    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape,
                                 int(cfg.replay_buffer_capacity),
                                 device)

    video_recorder = VideoRecorder(cfg.log_dir if cfg.save_video else None)

    # Initialize Progress Bar
    pbar = tqdm(total=cfg.num_train_steps, desc="Training Progress", unit="step")

    # Initialize Variables
    episode = 0
    episode_reward = 0
    done = True
    start_time = time.time()

    # Training Loop
    for step in range(1, cfg.num_train_steps + 1):
        if done:
            if step > 1:
                duration = time.time() - start_time
                logger.log('duration', duration, step, category='train')
                start_time = time.time()
                logger.dump(step, save=(step > cfg.num_seed_steps))

            if step > 0 and step % cfg.eval_frequency == 0:
                logger.log('episode', episode, step, category='eval')
                evaluate(step, env, agent, logger, video_recorder, cfg, pbar)

            logger.log('episode_reward', episode_reward, step, category='train')
            obs = env.reset()
            agent.reset()
            done = False
            episode_reward = 0
            episode += 1
            logger.log('episode', episode, step, category='train')

        # Select Action
        if step < cfg.num_seed_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.act(obs, sample=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)  # Ensure action is within bounds

        # Update Agent
        if step >= cfg.num_seed_steps:
            agent.update(replay_buffer, logger, step)

            # # Log additional metrics
            # try:
            #     q_values = agent.get_q_values()  # Implement this method in SACAgent
            #     policy_entropy = agent.get_policy_entropy()  # Implement this method in SACAgent
            #
            #     logger.log('q_value_mean', q_values.mean().item(), step, category='train')
            #     logger.log('policy_entropy', policy_entropy.item(), step, category='train')
            # except AttributeError as e:
            #     print(f"Agent method missing: {e}")

        # Take Step in Environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

        # Log Transitions with Confounder Info
        logger.log_transition(obs, action, reward, next_obs, done, step, confounder=cfg.confounders[cfg.env])

        # Update Observations
        obs = next_obs

        # Update Progress Bar
        pbar.update(1)

        # Log Additional Metrics at Specified Intervals
        if step % cfg.log_frequency == 0:
            log_additional_metrics(logger, step, pbar)

    # Finalize Training
    pbar.close()
    logger.dump(cfg.num_train_steps)
    logger.dump(cfg.num_train_steps, ty='transitions')
    if cfg.log_save_wandb:
        wandb.finish()
    print("Training completed successfully.")


def evaluate(step, env, agent, logger, video_recorder, cfg, pbar):
    print(f"Starting evaluation at step {step}")
    avg_reward = 0.0
    with tqdm(total=cfg.num_eval_episodes, desc="Evaluation", leave=False, unit="episode") as eval_pbar:
        for e in range(1, cfg.num_eval_episodes + 1):
            obs = env.reset()
            agent.reset()
            video_recorder.init(enabled=(e == 1))
            done = False
            ep_reward = 0
            while not done:
                with eval_mode(agent):
                    action = agent.act(obs, sample=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                video_recorder.record(env)
                ep_reward += reward
            avg_reward += ep_reward
            video_recorder.save(f'{step}_episode_{e}.mp4')
            eval_pbar.update(1)
    avg_reward /= cfg.num_eval_episodes

    logger.log('episode_reward', avg_reward, step, category='eval')
    logger.dump(step)

    # Update main progress bar with evaluation results
    pbar.set_postfix({'Eval Avg Reward': f"{avg_reward:.2f}"})
    print(f"Completed evaluation at step {step}, Avg Reward: {avg_reward:.2f}")


def log_additional_metrics(logger, step, pbar):
    # Retrieve rolling average metrics
    avg_reward = logger.get_average('episode_reward', category='train')
    critic_loss = logger.get_average('critic_loss', category='train')
    actor_loss = logger.get_average('actor_loss', category='train')
    alpha_loss = logger.get_average('alpha_loss', category='train')
    alpha_value = logger.get_average('alpha_value', category='train')

    # Update the progress bar's postfix with current metrics
    pbar.set_postfix({
        'Avg Reward': f"{avg_reward:.2f}",
        'Critic Loss': f"{critic_loss:.4f}",
        'Actor Loss': f"{actor_loss:.4f}",
        'Alpha Loss': f"{alpha_loss:.4f}",
        'Alpha Value': f"{alpha_value:.4f}"
    })


if __name__ == '__main__':
    main()
