#!/usr/bin/env python3
import os
import time
import hydra
from omegaconf import DictConfig
import torch
import dmc2gym
from utils.env_wrapper import DMCConfounderWrapper
from utils.replay_buffer import ReplayBuffer
from logger.logger import Logger
from utils.utils import set_seed_everywhere, eval_mode
from logger.video_recorder import VideoRecorder
from sac.sac import SACAgent
import numpy as np

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    def make_env(cfg):
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])
        env = DMCConfounderWrapper(domain_name=domain_name, task_name=task_name,
                                   seed=cfg.seed,
                                   confounder_params=cfg.confounder)
        return env

    # Setup
    work_dir = os.getcwd()
    print(f"workspace: {work_dir}")
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    env = make_env(cfg)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]

    # update cfg with dimensions
    cfg.agent.params.obs_dim = obs_dim
    cfg.agent.params.action_dim = action_dim
    cfg.agent.params.action_range = action_range
    cfg.agent.params.critic_cfg.obs_dim = obs_dim
    cfg.agent.params.critic_cfg.action_dim = action_dim
    cfg.agent.params.actor_cfg.obs_dim = obs_dim
    cfg.agent.params.actor_cfg.action_dim = action_dim

    wandb_config = None
    if cfg.log_save_wandb:
        wandb_config = {
            'project': 'SAC_DMC',
            'name': cfg.experiment,
            'config': dict(cfg),
            'reinit': True
        }

    logger = Logger(cfg.log_dir,
                    save_tb=cfg.log_save_tb,
                    log_frequency=cfg.log_frequency,
                    agent=cfg.agent.name,
                    save_wandb=cfg.log_save_wandb,
                    wandb_config=wandb_config)

    agent = SACAgent(**cfg.agent.params)

    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape,
                                 int(cfg.replay_buffer_capacity),
                                 device)

    video_recorder = VideoRecorder(cfg.log_dir if cfg.save_video else None)

    step = 0
    episode = 0
    episode_reward = 0
    done = True
    start_time = time.time()

    def evaluate(step):
        avg_reward = 0.0
        for e in range(cfg.num_eval_episodes):
            obs = env.reset()
            agent.reset()
            video_recorder.init(enabled=(e == 0))
            done = False
            ep_reward = 0
            while not done:
                with eval_mode(agent):
                    action = agent.act(obs, sample=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                video_recorder.record(env)
                ep_reward += reward
            avg_reward += ep_reward
            video_recorder.save(f'{step}.mp4')
        avg_reward /= cfg.num_eval_episodes
        logger.log('eval/episode_reward', avg_reward, step)
        logger.dump(step)

    while step < cfg.num_train_steps:
        if done:
            if step > 0:
                logger.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                logger.dump(step, save=(step > cfg.num_seed_steps))

            if step > 0 and step % cfg.eval_frequency == 0:
                logger.log('eval/episode', episode, step)
                evaluate(step)

            logger.log('train/episode_reward', episode_reward, step)
            obs = env.reset()
            agent.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            logger.log('train/episode', episode, step)

        if step < cfg.num_seed_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.act(obs, sample=True)

        if step >= cfg.num_seed_steps:
            agent.update(replay_buffer, logger, step)

        next_obs, reward, terminated, truncated, info = env.step(action)
        d = terminated or truncated
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

        # Log transitions with confounder info
        logger.log_transition(obs, action, reward, next_obs, d, step, confounder=cfg.confounder)

        obs = next_obs
        episode_step += 1
        step += 1

    logger.dump(step)
    logger.dump(step, ty='transitions')
    if cfg.log_save_wandb:
        import wandb
        wandb.finish()

if __name__ == '__main__':
    main()
