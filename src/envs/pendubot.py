import gym

from gym import spaces
from gym.utils import seeding

import numpy as np


class PendubotEnv(gym.Env):

    def __init__(
            self,
            factor=1.0,
            dt=0.01,
            control_scale=1,
    ):
        self._factor = factor
        self._dt = dt
        self._control_scale = control_scale
        self._max_control = 1.0 * self._control_scale * 2.0

        # Pendubot dynamics from Section V-A in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9345365
        # x1-theta1, x2-theta1dot, x3-theta2, x4-theta2dot
        # continuous-time dynamics xdot = Ac*x + Bc*u
        Ac = np.array([[0, 1, 0, 0],
                       [67.38, 0, -24.83, 0],
                       [0, 0, 0, 1],
                       [-69.53, 0, 105.32, 0]])
        Bc = np.array([[0],
                       [44.87],
                       [0],
                       [-85.09]]) / self._control_scale

        # discrete-time system
        self._AG = Ac * self._dt + np.eye(4)
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

        ts = 1
        self.x1lim = 1.0 * factor * ts
        self.x2lim = 2.0 * factor * ts * 10
        self.x3lim = 1.0 * factor * ts
        self.x4lim = 4.0 * factor * ts * 5  # xmax limits
        xmax = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.np_random = None
        self.seed()

        self._state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x1, x2, x3, x4 = self._state
        u0 = u
        u = np.clip(u, -self._max_control, self._max_control)[0]
        costs = 1 / self._factor ** 2 * (
                    1.0 * x1 ** 2 + 0.05 * x2 ** 2 + 1.0 * x3 ** 2 + 0.05 * x4 ** 2 + 0.2 * (
                        u / self._control_scale * self._factor) ** 2) - 5.0

        self._state = self._AG @ self._state + self._BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self._state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, seed=None):
        self.seed(seed)

        high = np.array([0.05, 0.1, 0.05, 0.1]) * self._factor / 1
        self._state = self.np_random.uniform(low=-high, high=high)
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


class PendubotEnvPartial(PendubotEnv):
    def __init__(
            self,
            factor=1,
            dt=0.01,
            control_scale=1,
    ):
        super(PendubotEnvPartial, self).__init__(
            factor=factor,
            dt=dt,
            control_scale=control_scale
        )
        self._obs_scale = 1
        self._CG = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]) * self._obs_scale

        # observations are the first state: pos
        xmax = np.array([self.x1lim, self.x3lim]) * self._obs_scale
        self.observation_space = spaces.Box(low=-xmax, high=xmax)

    def get_obs(self):
        return self._CG @ self._state


class PendubotEnvPartialNorm(PendubotEnvPartial):
    def __init__(
            self,
            factor=1,
            dt=0.01,
            control_scale=1,
    ):
        super(PendubotEnvPartialNorm, self).__init__(
            factor=factor,
            dt=dt,
            control_scale=control_scale
        )
        self._CG = self._CG / self.observation_space.high[:, np.newaxis]


if __name__ == '__main__':
    env = PendubotEnvPartialNorm()
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
