import gym

from gym import spaces
from gym.utils import seeding

import numpy as np


class InvPendulumNLEnv(gym.Env):

    def __init__(
            self, factor=1
    ):

        self.factor = factor
        self.viewer = None
        self.g = 10.0
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        self.max_torque = 2 #
        self.max_speed = 8.0 * factor
        self.max_pos = 1.5 * factor

        self._AG = np.array([
            [1, self.dt],
            [self.g/self.l*self.dt, 1-self.mu/(self.m*self.l**2)*self.dt]
        ])
        self._BG1 = np.array([[0], [-self.g*self.dt/self.l]]) #* factor
        self._BG2 = np.array([[0], [self.dt/(self.m*self.l**2)]]) * factor
        self._CG1 = np.array([[1, 0]])
        self._CG2 = np.array([[1, 0]])
        self._DG1 = np.array([[0]])

        # Delta = x1 - sin(x1) is sector bounded in [alpha_Delta, beta_Delta]
        alpha_Delta = 0.0
        beta_Delta = 0.41 # corresponds to the sector bound where x1 in [-1.4, 1.4]
        # filer Psi = [Dpsi1, Dpsi2]
        self._Dpsi1 = np.array([[beta_Delta],
                               [-alpha_Delta]])
        self._Dpsi2 = np.array([[-1],
                               [1]])
        # M matrix for IQC
        self._M = np.array([[0, 1],
                           [1, 0]])

        # dynamics of the extended system of G and Psi
        self._Ae = self._AG
        self._Be1 = self._BG1
        self._Be2 = self._BG2
        self._Ce1 = self._Dpsi1 @ self._CG1
        self._De1 = self._Dpsi1 @ self._DG1 + self._Dpsi2
        self._Ce2 = self._CG2

        # env setup
        self.npsi = 0
        self.nr = self._Dpsi1.shape[0]
        self.nx = self._AG.shape[0]
        self.nxe = self._Ae.shape[0]
        self.nq = self._BG1.shape[1]
        self.nu = self._BG2.shape[1]

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(self.nu,))
        # observations are the two states
        xmax = np.array([self.max_pos, self.max_speed])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.np_random = None
        self.seed()

        self._state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self._state

        g = self.g
        m = self.m
        l = self.l
        mu = self.mu
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = 1/self.factor**2*(th**2 + .1*thdot**2 + 1*((u*self.factor)**2)) - 1

        newthdot = thdot + (g/l*np.sin(th) - mu/(m*l**2)*thdot + 1/(m*l**2)*(u*self.factor)) * dt
        newth = th + thdot * dt

        self._state = np.array([newth, newthdot])

        terminated = False
        if self.time > 200 or not self.state_space.contains(self._state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, seed=None):
        self.seed(seed)

        high = np.array([np.pi/30, np.pi/20]) * self.factor
        self._state = self.np_random.uniform(low=-high, high=high)
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self._state / self.factor

    def get_state(self):
        return self._state


class InvPendulumNLEnvPartial(InvPendulumNLEnv):
    def __init__(self, factor=1):
        super().__init__(factor)
        self._CG2 = np.array([[1, 0]])

        # observations are the first state: pos
        xmax = np.array([self.max_pos, self.max_speed])
        self.observation_space = spaces.Box(low=-xmax[0:1], high=xmax[0:1])

    def get_obs(self):
        return self._CG2 @ self._state


class InvPendulumNLEnvPartialNorm(InvPendulumNLEnvPartial):
    def __init__(self, factor=1):
        super().__init__(factor)
        self._CG2 = self._CG2 / self.observation_space.high
