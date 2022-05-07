from .cartpole import CartpoleEnv
from .cartpole import CartpoleEnvPartial
from .cartpole import CartpoleEnvPartialNorm
from .pendulum import InvPendulumLinEnv
from .pendulum import InvPendulumLinEnvPartial
from .pendulum import InvPendulumLinEnvPartialNorm


def get_env(exp_name, factor):
    if 'cartpole' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = CartpoleEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = CartpoleEnvPartial(factor=factor)
        else:
            env = CartpoleEnv(factor=factor)
    elif 'pendulum' in exp_name:
        if 'partial' in exp_name and 'norm' in exp_name:
            env = InvPendulumLinEnvPartialNorm(factor=factor)
        elif 'partial' in exp_name:
            env = InvPendulumLinEnvPartial(factor=factor)
        else:
            env = InvPendulumLinEnv(factor=factor)
    else:
        raise ValueError('Unrecognized environment in experiment name: ' + exp_name)

    return env
