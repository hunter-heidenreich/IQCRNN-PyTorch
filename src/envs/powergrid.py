import gym
import torch

from gym import spaces
from gym.utils import seeding

import numpy as np


class PowergridEnv(gym.Env):

    def __init__(
            self,
            factor=1,
            dt=0.2,
            control_scale=1,
    ):
        self._factor = factor
        self.viewer = None
        self._dt = dt  # sampling time

        # maximum control input
        self._control_scale = control_scale
        self._max_control = 1.0 * self._control_scale * 2

        # powergrid dynamics in Section IV-B of https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9296215
        M = np.diag([4.0000, 3.0000, 2.5000, 4.0000, 2.0000, 3.5000, 3.0000, 2.5000, 2.0000, 6.0000])
        L = np.array([
            [33.1348,   -2.5425,   -2.8928,   -2.0816,   -0.9371,   -2.1775,   -1.2132,  -10.3953,   -3.1871,   -7.7078],
            [-2.5425,   26.4495,  -10.5043,   -1.6747,   -0.7539,   -1.7518,   -0.9760,   -1.3576,   -0.8933,   -5.9955],
            [-2.8928,  -10.5043,   28.2182,   -2.3262,   -1.0471,   -2.4333,   -1.3557,   -1.5863,   -1.1153,   -4.9572],
            [-2.0816,   -1.6747,   -2.3262,   34.5327,  -16.1015,   -5.1567,   -2.8730,   -1.4638,   -1.5682,   -1.2871],
            [-0.9371,   -0.7539,   -1.0471,  -16.1015,   24.3984,   -2.3213,   -1.2933,   -0.6589,   -0.7059,   -0.5794],
            [-2.1775,   -1.7518,   -2.4333,   -5.1567,   -2.3213,   32.5848,  -14.2263,   -1.5312,   -1.6404,   -1.3463],
            [-1.2132,   -0.9760,   -1.3557,   -2.8730,   -1.2933,  -14.2263,   24.4547,   -0.8531,   -0.9139,   -0.7501],
            [-10.3953,   -1.3576,   -1.5863,   -1.4638,   -0.6589,   -1.5312,   -0.8531,   25.7938,   -4.4344,   -3.5132],
            [-3.1871,   -0.8933,   -1.1153,   -1.5682,   -0.7059,   -1.6404,   -0.9139,   -4.4344,   15.7360,   -1.2775],
            [-7.7078,   -5.9955,   -4.9572,   -1.2871,   -0.5794,   -1.3463,   -0.7501,   -3.5132,   -1.2775,   27.4140],
        ])
        D = np.diag([5.0000, 4.0000, 4.0000, 6.0000, 3.5000, 3.0000, 7.5000, 4.0000, 6.5000, 5.0000])

        # continuous-time dynamics xdot = Ac*x + Bc*u
        Ac = np.block([
            [np.zeros((10, 10)), np.eye(10)],
            [-np.linalg.inv(M) @ L, -np.linalg.inv(M) @ D],
        ])
        Bc = np.block([[np.zeros((10, 10))], [np.linalg.inv(M)]]) / self._control_scale

        # discrete-time system
        self._AG = Ac * self._dt + np.eye(20)
        self._BG = Bc * self._dt * factor

        self._CG = None

        self.nx = self._AG.shape[0]
        self.nu = self._BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(
            low=-self._max_control,
            high=self._max_control,
            shape=(self.nu,)
        )

        self.thetalim = 0.5 * factor * np.ones(10,)
        self.omegalim = 1.2 * factor * np.ones(10,)
        xmax = np.block([self.thetalim, self.omegalim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.np_random = None
        self.seed()

        self._state = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        # u = u.detach().numpy()
        u = np.clip(u, -self._max_control, self._max_control)
        costs = 1/self._factor**2 * (np.linalg.norm(self._state, 2)**2 + 0.2 * np.linalg.norm(u / self._control_scale * self._factor, 2)**2) - 5.0

        self._state = self._AG @ self._state + self._BG @ u.numpy()

        terminated = False
        if self.time > 200 or not self.state_space.contains(self._state):
            terminated = True

        self.time += 1

        costs = torch.tensor(costs, dtype=torch.float32)

        return self.get_obs(), -costs, terminated, {}

    def reset(self, seed=None):
        self.seed(seed)

        high = np.block([self.thetalim / 10, self.omegalim / 10]) * self._factor
        self._state = self.np_random.uniform(low=-high, high=high)
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self._state

    def get_state(self):
        return self._state

    def get_system_params(self):
        return self._AG, self._BG, self._CG


class PowergridEnvPartial(PowergridEnv):
    def __init__(
            self,
            factor=1,
            dt=0.2,
            control_scale=1,
    ):
        super().__init__(
            factor=factor,
            dt=dt,
            control_scale=control_scale,
        )
        self._obs_scale = 1
        self._CG = np.block([
            np.eye(10), np.zeros((10, 10))
        ]) * self._obs_scale

        # observations are the first 10 state: theta1 to theta10
        xmax = self.thetalim * self._obs_scale
        self.observation_space = spaces.Box(low=-xmax, high=xmax)

    def get_obs(self):
        return self._CG @ self._state


class PowergridEnvPartialNorm(PowergridEnvPartial):
    def __init__(
            self,
            factor=1,
            dt=0.2,
            control_scale=1,
    ):
        super().__init__(
            factor=factor,
            dt=dt,
            control_scale=control_scale,
        )
        self._CG = self._CG / self.observation_space.high[:, np.newaxis]