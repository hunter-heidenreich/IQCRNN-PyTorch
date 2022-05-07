import gym

from gym import spaces
from gym.utils import seeding

import numpy as np


class InvPendulumLinEnv(gym.Env):
    """

    """

    def __init__(
            self,
            g=10.0,
            m=0.15,
            l=0.5,
            mu=0.05,
            factor=1.0,
            max_speed=8.0,
            max_torque=2.0,
            max_pos=1.5,
            dt=0.02,
    ):
        self._factor = factor
        self._g = g
        self._m = m
        self._l = l
        self._mu = mu

        self._max_speed = max_speed * self._factor
        self._max_torque = max_torque
        self._max_pos = max_pos * self._factor

        self._dt = dt

        # Plant state-space
        self._AG = np.array([
            [1,                                 self._dt],
            [(self._g / self._l) * self._dt,    1 - (self._mu/(self._m * self._l**2)) * self._dt]
        ], dtype=np.float32)
        self._BG = np.array([
            [0],
            [self._dt/(self._m * self._l**2)]
        ], dtype=np.float32) * self._factor
        self._CG = None  # should be over-ridden in derived observables

        self.nx = self._AG.shape[0]
        self.nu = self._BG.shape[1]

        self.time = 0

        self.clock = None
        self.window = None

        self.action_space = spaces.Box(
            low=-self._max_torque,
            high=self._max_torque,
            shape=(self.nu,))

        # observations are the two states
        xmax = np.array([self._max_pos, self._max_speed])
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
        th, thdot = self._state

        # Essentially a "sat" function
        # clips input to +/- max torque that can be applied
        u = np.clip(u, -self._max_torque, self._max_torque)[0]

        # reward (aka negative cost)
        costs = 1 / self._factor ** 2 * (th ** 2 + .1 * thdot ** 2 + 1 * (
                    (u * self._factor) ** 2)) - 1

        self._state = self._AG @ self._state + self._BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self._state):
            terminated = True

        self.time += 1
        self._last_u = u

        return self.get_obs(), -costs, terminated, {}

    def reset(self, seed=None):
        self.seed(seed)

        high = np.array([np.pi / 30, np.pi / 20]) * self._factor
        self._state = self.np_random.uniform(low=-high, high=high).astype(np.float32)

        self._last_u = None
        self.time = 0

        return self.get_obs()

    def get_state(self):
        return self._state  # / self._factor

    def get_obs(self):
        return self._state / self._factor

    def render(self, mode="human"):
        raise NotImplementedError()

    def get_system_params(self):
        return self._AG, self._BG, self._CG


class InvPendulumLinEnvPartial(InvPendulumLinEnv):
    def __init__(
            self,
            g=10.0,
            m=0.15,
            l=0.5,
            mu=0.05,
            factor=1.0,
            max_speed=8.0,
            max_torque=2.0,
            max_pos=1.5,
            dt=0.02,
    ):
        super().__init__(
            g=g,
            m=m,
            l=l,
            mu=mu,
            factor=factor,
            max_speed=max_speed,
            max_torque=max_torque,
            max_pos=max_pos,
            dt=dt,
        )

        # Update observation matrix...
        # Now, only angular position is observable
        self._CG = np.array([[1, 0]], dtype=np.float32)

        # observations are the first state: pos
        # update this space
        xmax = np.array([self._max_pos, self._max_speed])
        self.observation_space = spaces.Box(low=-xmax[0:1], high=xmax[0:1])

    def get_obs(self):
        return self._CG @ self._state


class InvPendulumLinEnvPartialNorm(InvPendulumLinEnvPartial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # re-scale observation --> Now observables will be in [-1, 1]
        self._CG = self._CG / self.observation_space.high


if __name__ == '__main__':
    env = InvPendulumLinEnvPartialNorm(factor=1e-1)
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
