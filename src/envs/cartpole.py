import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class CartpoleEnv(gym.Env):

    def __init__(
            self,
            factor=1.0,
            dt=0.02,
            max_control=2,
    ):

        self._factor = factor
        self._viewer = None
        self._dt = dt

        # maximum control input
        self._max_control = max_control

        # discrete-time model for control synthesis
        self._AG = np.array([
            [1.0, -0.001, 0.02, 0.0],
            [0.0, 1.005, 0.0, 0.02],
            [0.0, -0.079, 1.0, -0.001],
            [0.0, 0.550, 0.0, 1.005]
        ], dtype=np.float32)
        # self.BG = np.array([
        #    [0.0], [0.0], [0.008], [-0.008]
        # ]) * self._factor
        self._BG = np.array([
            [0.0], [0.0], [0.04], [-0.04]
        ], dtype=np.float32) * self._factor
        self._CG = None

        self.nx = self._AG.shape[0]
        self.nu = self._BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(
            low=-self._max_control,
            high=self._max_control,
            shape=(self.nu,)
        )

        ts = 1  # testing scale
        self.x1lim = 1.0 * self._factor * ts
        self.x2lim = np.pi / 2 * self._factor * ts
        self.x3lim = 5.0 * self._factor * ts
        self.x4lim = 2.0 * np.pi * self._factor * ts

        # xmax limits
        xmax = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])

        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.np_random = None
        self.seed()

        self._state = None
        self._last_u = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # Un-pack state
        x1, x2, x3, x4 = self._state

        # Sat. function
        u = np.clip(u, -self._max_control, self._max_control)[0]

        # negative reward
        costs = 1/self._factor**2 * (1.0 * x1**2 + 1.0 * x2**2 + 0.04 * x3**2 + 0.1 * x4**2 + 0.2 * (u * self._factor)**2) - 5.0

        self._state = self._AG @ self._state + self._BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self._state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, seed=None):
        self.seed(seed)

        # high = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim]) / 2.0 / 10
        # xlim 1, pi/2, 5, pi
        high = np.array([0.05, 0.05, 0.25, 0.15]) # / 4 / 4
        self._state = self.np_random.uniform(low=-high, high=high)
        self._last_u = None
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return self._state

    def get_state(self):
        return self._state

    def render(self, mode="human"):
        raise NotImplementedError()

    def get_system_params(self):
        return self._AG, self._BG, self._CG


class CartpoleEnvPartial(CartpoleEnv):

    def __init__(
            self,
            factor=1.0,
            dt=0.02,
            max_control=2,
    ):
        super(CartpoleEnvPartial, self).__init__(
            factor=factor,
            dt=dt,
            max_control=max_control,
        )

        self._CG = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # observations are the first state: pos
        xmax = np.array([self.x1lim, self.x2lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)

    def get_obs(self):
        return self._CG @ self._state


class CartpoleEnvPartialNorm(CartpoleEnvPartial):

    def __init__(
            self,
            factor=1.0,
            dt=0.02,
            max_control=2,
    ):
        super(CartpoleEnvPartialNorm, self).__init__(
            factor=factor,
            dt=dt,
            max_control=max_control,
        )

        self._CG = self._CG / self.observation_space.high[:, np.newaxis]


if __name__ == '__main__':
    env = CartpoleEnvPartialNorm()
    print(env)
    print('Action space:', env.action_space)
    print('Observation space:', env.observation_space)
    print('State space:', env.state_space)

    obs = env.reset(seed=42)

    for _ in range(200):
        obs, reward, done, info = env.step(env.action_space.sample())
        st = env.get_state()
        # env.render()
        print(env.time, obs, obs*env.observation_space.high, st, reward, done)

        if done:
            break
        #     observation, info = env.reset(return_info=True)

    env.close()