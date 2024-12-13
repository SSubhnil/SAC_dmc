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
                 confounder_params=None, action_masking_params=None):
        self._from_pixels = from_pixels
        self.confounder_params = confounder_params or {}
        self.action_masking_params = action_masking_params or {}
        self.masked_joints_ids = []  # To store joint IDs for masking
        self.force_application_enabled = False
        self.force_type = 'step'
        self.force_range = [10.0, 20.0]
        self.interval_mean = 100
        self.interval_std = 10
        self.random_chance = 0.5
        self.duration_min = 10
        self.duration_max = 20
        self.body_part_to_apply_force = 'torso'
        self.timing = 'fixed'  # or 'random'
        self.time_since_last_force = 0
        self.interval = self.interval_mean  # Initialize interval

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

        # Apply confounders
        self.apply_confounders()

        # Apply action masking
        self.apply_action_masking()

    def apply_confounders(self):
        # Apply weight scaling
        weight_conf = self.confounder_params.get('weight', {})
        scale_factor = weight_conf.get('scale_factor', 1.0)
        body_parts = weight_conf.get('body_parts', {})

        for body_part, scale in body_parts.items():
            try:
                body_id = self._env.physics.model.name2id(body_part, 'body')
                original_mass = self._env.physics.model.body_mass[body_id]
                self._env.physics.model.body_mass[body_id] = original_mass * scale_factor * scale
            except KeyError:
                print(f"Warning: Body part '{body_part}' not found in the environment.")

        # Apply friction scaling
        friction_conf = self.confounder_params.get('friction', {})
        friction_scale = friction_conf.get('scale_factor', 1.0)
        try:
            geom_id = self._env.physics.model.name2id('floor', 'geom')
            self._env.physics.model.geom_friction[geom_id, :] *= friction_scale
        except KeyError:
            print("Warning: 'floor' geom not found in the environment.")

        # Apply gravity scaling
        gravity_conf = self.confounder_params.get('gravity', {})
        gravity_scale = gravity_conf.get('scale_factor', 1.0)
        self._env.physics.model.opt.gravity[:] *= gravity_scale

        # Handle external_force
        external_force_conf = self.confounder_params.get('external_force', {})
        self.force_application_enabled = external_force_conf.get('enabled', False)
        if self.force_application_enabled:
            self.force_type = external_force_conf.get('force_type', 'step')
            self.force_range = external_force_conf.get('force_range', [10.0, 20.0])
            self.interval_mean = external_force_conf.get('interval_mean', 100)
            self.interval_std = external_force_conf.get('interval_std', 10)
            self.random_chance = external_force_conf.get('random_chance', 0.5)
            self.duration_min = external_force_conf.get('duration_min', 10)
            self.duration_max = external_force_conf.get('duration_max', 20)
            self.body_part_to_apply_force = external_force_conf.get('body_part', 'torso')
            self.timing = 'random' if self.interval_std > 0 else 'fixed'
            self.time_since_last_force = 0
            self.interval = self.interval_mean

    def apply_action_masking(self):
        if self.action_masking_params.get('enabled', False):
            mode = self.action_masking_params.get('mode', 'joint')
            cripple_mode = self.action_masking_params.get('cripple_mode', 'whole_leg')
            cripple_targets = self.action_masking_params.get('cripple_targets', [])
            cripple_joints = []

            if cripple_mode == 'whole_leg':
                # Define mapping from leg to joints
                leg_to_joints = {
                    'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
                    'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
                    'front_leg': ['front_hip', 'front_knee', 'front_ankle'],
                    'rear_leg': ['back_hip', 'back_knee', 'back_ankle'],
                    'front_left_leg': ['front_left_hip', 'front_left_knee', 'front_left_ankle'],
                    'front_right_leg': ['front_right_hip', 'front_right_knee', 'front_right_ankle'],
                    'rear_left_leg': ['rear_left_hip', 'rear_left_knee', 'rear_left_ankle'],
                    'rear_right_leg': ['rear_right_hip', 'rear_right_knee', 'rear_right_ankle']
                }
                for leg in cripple_targets:
                    joints = leg_to_joints.get(leg, [])
                    cripple_joints.extend(joints)
            elif cripple_mode == 'individual_joints':
                cripple_joints = self.action_masking_params.get('joints', [])

            # Add joints to mask
            for joint in cripple_joints:
                try:
                    joint_id = self._env.physics.model.name2id(joint, 'joint')
                    self.masked_joints_ids.append(joint_id)
                except KeyError:
                    print(f"Warning: Joint '{joint}' not found in the environment.")

    def apply_force(self):
        if self.timing == 'random':
            self.interval = max(30, int(np.random.normal(self.interval_mean,
                                                         self.interval_std)))
            if np.random.uniform() > self.random_chance:
                return

        # Update the timing
        self.time_since_last_force += 1
        if self.time_since_last_force < self.interval:
            return

        # Reset timing for next force application
        self.time_since_last_force = 0

        # Sample the force magnitude from a normal distribution within the range
        force_magnitude = np.clip(np.random.normal((self.force_range[0] + self.force_range[1]) / 2,
                                                   (self.force_range[1] - self.force_range[0]) / 6),
                                  self.force_range[0], self.force_range[1])

        # Calculate the duration for the force application if 'swelling'
        duration = np.random.randint(self.duration_min, self.duration_max + 1)

        # Flipping the direction for additional challenge
        direction = np.random.choice([-1, 1])

        # Apply swelling or other dynamics based on force type
        # Construct the force vector
        if self.force_type == 'step':
            force = np.array([direction * force_magnitude, 0, 0, 0, 0, 0])
        elif self.force_type == 'swelling':
            # Calculate the time step where the force magnitude is at its peak
            peak_time = duration / 2
            # Calculate the standard deviation to control the width of the bell curve
            sigma = duration / 6  # Adjust as needed for the desired width
            # Calculate the force magnitude at the current time step using a Gaussian function
            time_step_normalized = (self.time_since_last_force - peak_time) / sigma
            magnitude = force_magnitude * np.exp(-0.5 * (time_step_normalized ** 2))
            force = np.array([direction * magnitude, 0, 0, 0, 0, 0])

        try:
            body_id = self._env.physics.model.name2id(self.body_part_to_apply_force, 'body')
            # Apply the force
            self._env.physics.data.xfrc_applied[body_id] = force
        except KeyError:
            print(f"Warning: Body part '{self.body_part_to_apply_force}' not found in the environment.")

    def _action_mask(self, action):
        # Zero out actions for masked joints
        for joint_id in self.masked_joints_ids:
            action[joint_id] = 0.0  # Assuming joint actions are indexed accordingly
        return action

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
        # Apply external force if enabled
        if self.force_application_enabled:
            self.apply_force()

        # Apply action masking
        action = self._action_mask(action)

        # Clip action to ensure validity
        action = np.clip(action, self._true_action_space.low, self._true_action_space.high).astype(np.float32)

        # Step the environment
        time_step = self._env.step(action)
        reward = time_step.reward or 0
        terminated = time_step.last()
        obs = _flatten_obs(time_step.observation)
        truncated = False  # Adjust based on your environment's truncation criteria
        return obs, reward, terminated, truncated, {}

    def reset(self):
        time_step = self._env.reset()
        obs = _flatten_obs(time_step.observation)
        return obs

    def render(self, mode='rgb_array'):
        return self._env.physics.render(height=240, width=320, camera_id=0)
