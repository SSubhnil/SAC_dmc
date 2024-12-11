import gymnasium as gym
from gymnasium import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np

def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype in [np.float64, np.float32]
        dim = int(np.prod(s.shape))
        if isinstance(s, specs.Array):
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif isinstance(s, specs.BoundedArray):
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    return spaces.Box(low, high, dtype=dtype)

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCConfounderWrapper(core.Env):
    def __init__(self, domain_name, task_name, seed=0, from_pixels=False,
                 confounder_params=None):
        self._from_pixels = from_pixels
        self.confounder_params = confounder_params or {}

        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={'random': seed},
            visualize_reward=True
        )

        # true action space
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        # normalized action space [-1,1]
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0,
                                             shape=self._true_action_space.shape,
                                             dtype=np.float32)

        # observation space
        self._observation_space = _spec_to_box(self._env.observation_spec().values(),
                                               np.float64)

        self.seed(seed)

        # apply confounders
        self.apply_confounders()

    def apply_confounders(self):
        # Modify environment parameters as per confounder_params
        # body_weights, friction, gravity, external_force, cripple_part
        if 'body_weights' in self.confounder_params:
            for body_name, weight in self.confounder_params['body_weights'].items():
                body_id = self._env.physics.model.name2id(body_name, 'body')
                self._env.physics.model.body_mass[body_id] = weight

        if 'friction' in self.confounder_params:
            friction_value = self.confounder_params['friction']
            # Assume floor geom name is 'floor'
            geom_id = self._env.physics.model.name2id('floor', 'geom')
            self._env.physics.model.geom_friction[geom_id, :] = [friction_value, 0.005, 0.0001]

        if 'gravity' in self.confounder_params and self.confounder_params['gravity'] is not None:
            self._env.physics.model.opt.gravity[:] = self.confounder_params['gravity']

        # cripple_part, external_force could be applied similarly if implemented

    def _convert_action(self, action):
        # Since action is already scaled to [low, high], just clip to ensure validity
        action = np.clip(action, self._true_action_space.low, self._true_action_space.high)
        return action.astype(np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        # DM Control seeding handled by task_kwargs
        pass

    def step(self, action):
        action = self._convert_action(action)
        reward = 0
        time_step = self._env.step(action)
        reward += time_step.reward or 0
        terminated = time_step.last()
        obs = _flatten_obs(time_step.observation)
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self):
        time_step = self._env.reset()
        obs = _flatten_obs(time_step.observation)
        return obs

    def render(self, mode='rgb_array'):
        return self._env.physics.render(height=240, width=320, camera_id=0)
