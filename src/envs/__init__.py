from .cartpole import CartpoleEnv
from .cartpole import CartpoleEnvPartial
from .cartpole import CartpoleEnvPartialNorm

from .pendubot import PendubotEnv
from .pendubot import PendubotEnvPartial
from .pendubot import PendubotEnvPartialNorm

from .pendulum import InvPendulumLinEnv
from .pendulum import InvPendulumLinEnvPartial
from .pendulum import InvPendulumLinEnvPartialNorm

from .pendulumNL import InvPendulumNLEnv
from .pendulumNL import InvPendulumNLEnvPartial
from .pendulumNL import InvPendulumNLEnvPartialNorm

from .powergrid import PowergridEnv
from .powergrid import PowergridEnvPartial
from .powergrid import PowergridEnvPartialNorm

from .vehicle import VehliceEnv
from .vehicle import VehliceEnvPartial
from .vehicle import VehliceEnvPartialNorm


def get_env(exp_name, factor):
    if 'cartpole' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = CartpoleEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = CartpoleEnvPartial(factor=factor)
        else:
            env = CartpoleEnv(factor=factor)
    elif 'pendulumNL' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = InvPendulumNLEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = InvPendulumNLEnvPartial(factor=factor)
        else:
            env = InvPendulumNLEnv(factor=factor)
    elif 'pendulum' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = InvPendulumLinEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = InvPendulumLinEnvPartial(factor=factor)
        else:
            env = InvPendulumLinEnv(factor=factor)
    elif 'pendubot' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = PendubotEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = PendubotEnvPartial(factor=factor)
        else:
            env = PendubotEnv(factor=factor)
    elif 'powergrid' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = PowergridEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = PowergridEnvPartial(factor=factor)
        else:
            env = PowergridEnv(factor=factor)
    elif 'vehicle' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = VehliceEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = VehliceEnvPartial(factor=factor)
        else:
            env = VehliceEnv(factor=factor)
    else:
        raise ValueError('Unrecognized environment in experiment name: ' + exp_name)

    return env
