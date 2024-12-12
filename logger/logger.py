import os
import shutil
import wandb
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
import csv
import json

from omegaconf import OmegaConf, DictConfig  # Import OmegaConf
from termcolor import colored

COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('duration', 'D', 'time')
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float')
]

AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float')
    ]
}

TRANSITION_FORMAT = [
    ('state', 'S', 'str'),
    ('action', 'A', 'str'),
    ('reward', 'Re', 'float'),
    ('next_state', 'NS', 'str'),
    ('done', 'D', 'int'),
    ('confounder', 'C', 'str')
]

class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0
    def update(self, value, n=1):
        self._sum += value*n
        self._count += n
    def value(self):
        return self._sum / max(1, self._count)

class TransitionsGroup:
    def __init__(self, file_name, format):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._format = format
        self._csv_file = open(self._csv_file_name, 'w', newline='')
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[f[0] for f in self._format])
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log_batch(self, data_dict):
        self._csv_writer.writerow(data_dict)
        self._csv_file.flush()

    def dump(self, step, prefix, save=True):
        # Transitions are logged per batch; no need to dump
        pass


class MetersGroup:
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w', newline='')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        if isinstance(value, str):
            # Skip logging string metrics
            return
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            val = meter.value()
            data[key] = val
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def dump(self, step, prefix, save=True):
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
        self._meters.clear()

class Logger:
    def __init__(self, log_dir, save_tb=False, log_frequency=10000, agent='sac', wandb_run=None):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        self._wandb_run = wandb_run

        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None

        # For rolling averages
        self.metrics = defaultdict(lambda: deque(maxlen=100))  # Store last 100 values for each metric

        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'), formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'), formating=COMMON_EVAL_FORMAT)
        self._transitions_group = TransitionsGroup(os.path.join(log_dir, 'transitions'), TRANSITION_FORMAT)

    def _should_log(self, step, log_frequency):
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step, category='train'):
        if self._sw is not None:
            if category == 'train':
                self._sw.add_scalar(f"train/{key}", value, step)
            elif category == 'eval':
                self._sw.add_scalar(f"eval/{key}", value, step)

    def log(self, key, value, step, n=1, log_frequency=1, category='train'):
        if not self._should_log(step, log_frequency):
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        if self._wandb_run is not None:
            wandb.log({key: value}, step=step)
        self._try_sw_log(key, value / n, step, category=category)
        if category == 'train':
            mg = self._train_mg
        elif category == 'eval':
            mg = self._eval_mg
        else:
            raise ValueError(f"Unknown category: {category}")
        mg.log(key, value, n)
        # Update rolling averages
        self.metrics[f"{category}/{key}"].append(value)

    def get_average(self, key, category='train'):
        metric_key = f"{category}/{key}"
        if metric_key in self.metrics and len(self.metrics[metric_key]) > 0:
            return sum(self.metrics[metric_key]) / len(self.metrics[metric_key])
        return 0.0

    def log_transition(self, state, action, reward, next_state, done, step, confounder=None, log_frequency=1000):
        if not self._should_log(step, log_frequency):
            return
        # Convert DictConfig to dict if necessary
        if isinstance(confounder, DictConfig):
            confounder = OmegaConf.to_container(confounder, resolve=True)
        data = {
            'state': json.dumps(state.tolist()),
            'action': json.dumps(action.tolist()),
            'reward': reward,
            'next_state': json.dumps(next_state.tolist()),
            'done': int(done),
            'confounder': json.dumps(confounder) if confounder else json.dumps({})
        }
        self._transitions_group.log_batch(data)
        if self._wandb_run is not None:
            # Log a subset to wandb to avoid huge data
            if np.random.rand() < 0.01:
                wandb.log({'transition': data}, step=step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
            self._transitions_group.dump(step, 'transitions', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        elif ty == 'transitions':
            self._transitions_group.dump(step, 'transitions', save)
