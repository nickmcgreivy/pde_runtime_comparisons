import jax.numpy as np
from flux import Flux, NetworkOutput
from flax.training import train_state  # Useful dataclass to keep train state
import argparse
import jax
import optax
from flax import serialization
from optax import polynomial_schedule
from rungekutta import ssp_rk3
import jax.random as random
from basisfunctions import num_elements

from arguments import get_args
from helper import f_to_FE
from model import LearnedStencil2D

PI = np.pi


def get_model(args, order):
    return LearnedStencil2D(
        features=[args.width for _ in range(args.depth)],
        kernel_size=args.kernel_size,
        kernel_out=args.kernel_out,
        width_even=args.width_even,
        width_odd=args.width_odd,
        order=order,
    )

def save_training(params, losses, directory, unique_id, order, up):
    bytes_output = serialization.to_bytes(params)
    with open(
        "{}/{}_order{}_up{}_params".format(directory, unique_id, order, up), "wb"
    ) as f:
        f.write(bytes_output)
    with open(
        "{}/{}_order{}_up{}_losses.txt".format(directory, unique_id, order, up), "w"
    ) as f:
        for listitem in losses:
            f.write("%s\n" % listitem)


def load_training(args, directory, unique_id, order, up):
    with open(
        "{}/{}_order{}_up{}_params".format(directory, unique_id, order, up), "rb"
    ) as f:
        param_bytes = f.read()
    with open(
        "{}/{}_order{}_up{}_losses.txt".format(directory, unique_id, order, up), "r"
    ) as f:
        losses = f.readlines()
        losses = [float(x.strip()) for x in losses]

    model = get_model(args, order)
    params = serialization.from_bytes(
        init_params(jax.random.PRNGKey(0), model, order), param_bytes
    )
    return model, params, losses


def init_params(key, model, order):
    NX_NO_MEANING = 128  # params doesn't depend on this
    NY_NO_MEANING = 128
    return model.init(
        key, np.zeros((NX_NO_MEANING, NY_NO_MEANING, num_elements(order)))
    )

def create_train_state(key, args, model, order):
    params = init_params(key, model, order)
    schedule_fn = polynomial_schedule(
        init_value=args.learning_rate,
        end_value=args.learning_rate / args.learning_rate_decay,
        power=1,
        transition_steps=1,
        transition_begin=args.training_iterations // 2,
    )
    if args.optimizer.lower() == "adam":
        tx = optax.chain(optax.adam(schedule_fn), optax.zero_nans(), optax.clip(0.1))
    elif args.optimizer.lower() == "sgd":
        tx = optax.chain(optax.sgd(schedule_fn), optax.zero_nans(), optax.clip(0.1))
    else:
        raise ValueError("Incorrect Optimizer")
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_initial_condition(key, args):
    def zeros(x, y, t):
        assert x.shape == y.shape
        return np.zeros(x.shape)

    def gaussian(x, y, t):
        xc, yc = args.Lx / 2, args.Ly / 2
        return np.exp(
            -75 * ((x - xc) ** 2 / args.Lx ** 2 + (y - yc) ** 2 / args.Ly ** 2)
        )

    def diffusion_test(x, y, t):
        return np.sin(x * 2 * PI / args.Lx) * np.cos(y * 2 * PI / args.Ly)

    """
    def gaussian_hw(x, y, t):
        xc, yc = args.Lx / 2, args.Ly / 2
        return np.exp(
            -400 * ((x - xc) ** 2 / args.Lx ** 2 + (y - yc) ** 2 / args.Ly ** 2)
        )
    """

    def cosine_hump(x, y, t):
        x0 = 0.25 * args.Lx
        y0 = 0.5 * args.Ly
        r0 = 0.2 * np.sqrt((args.Lx ** 2 + args.Ly ** 2) / 2)
        r = np.minimum(np.sqrt((x - x0) ** 2 + (y - y0) ** 2), r0) / r0
        return 0.25 * (1 + np.cos(np.pi * r))

    def two_cosine_humps(x, y, t):
        x0a = 0.25 * args.Lx
        x0b = 0.75 * args.Lx
        y0 = 0.5 * args.Ly
        r0 = 0.2 * np.sqrt((args.Lx ** 2 + args.Ly ** 2) / 2)
        ra = np.minimum(np.sqrt((x - x0a) ** 2 + (y - y0) ** 2), r0) / r0
        rb = np.minimum(np.sqrt((x - x0b) ** 2 + (y - y0) ** 2), r0) / r0
        return 0.25 * (1 + np.cos(np.pi * ra)) + 0.25 * (1 + np.cos(np.pi * rb))

    def double_shear(x, y, t):
        rho = 1 / 30
        delta = 0.05
        div = np.pi / 15
        return (
            delta * np.cos(2 * np.pi * x / args.Lx)
            + (y > args.Ly / 2) * np.cosh((3 / 4 - y / args.Ly) / rho) ** (-2) / div
            - (y <= args.Ly / 2) * np.cosh((y / args.Ly - 1 / 4) / rho) ** (-2) / div
        )

    def vortex_waltz(x, y, t):
        x1 = 0.35 * args.Lx
        y1 = 0.5 * args.Ly
        x2 = 0.65 * args.Lx
        y2 = 0.5 * args.Ly
        denom_x = 0.8 * (args.Lx ** 2) / (100.0)
        denom_y = 0.8 * (args.Ly ** 2 / 100)
        gaussian_1 = np.exp(-((x - x1) ** 2 / denom_x + (y - y1) ** 2 / denom_y))
        gaussian_2 = np.exp(-((x - x2) ** 2 / denom_x + (y - y2) ** 2 / denom_y))
        return gaussian_1 + gaussian_2


    def sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y):
        return np.sum(
            amplitudes[None, :]
            * np.sin(
                ks_x[None, :] * 2 * PI / args.Lx * x[:, None] + phases_x[None, :]
            ) * np.sin(
                ks_y[None, :] * 2 * PI / args.Ly * y[:, None] + phases_y[None, :]
            ),
            axis=1,
        )

    if args.initial_condition == "zero" or args.initial_condition == "zeros":
        return zeros
    elif args.initial_condition == "gaussian" and (
        args.equation == "advection" or args.equation == "euler"
    ):
        return gaussian
    elif args.initial_condition == "gaussian" and (
        args.equation == "hw" or args.equation == "hasegawa_wakatani"
    ):
        return gaussian
    elif args.initial_condition == "cosine_hump":
        return cosine_hump
    elif (
        args.initial_condition == "two_humps"
        or args.initial_condition == "two_cosine_humps"
    ):
        return two_cosine_humps
    elif args.initial_condition == "double_shear":
        return double_shear
    elif args.initial_condition == "vortex_waltz":
        return vortex_waltz
    elif args.initial_condition == "random":
        key1, key2, key3, key4, key5 = random.split(key, 5)
        phases_x = random.uniform(key1, (args.num_init_modes,)) * 2 * PI
        phases_y = random.uniform(key2, (args.num_init_modes,)) * 2 * PI
        ks_x = random.randint(
            key3, (args.num_init_modes,), args.min_k, args.max_k
        )
        ks_y = random.randint(
            key4, (args.num_init_modes,), args.min_k, args.max_k
        )
        amplitudes = random.uniform(key5, (args.num_init_modes,)) * args.amplitude_max
        return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)
    elif args.initial_condition == 'diffusion_test':
        return diffusion_test
    else:
        raise NotImplementedError


def get_f_phi(key, args, nx, ny, order):
    # returns numpy array of shape (nx, ny, num_FE)
    assert args.equation == "advection"
    phi_func = lambda func, t: f_to_FE(nx, ny, args.Lx, args.Ly, order, func, t)

    def zeros_func(x, y, t):
        return 0.0

    def right_func(x, y, t):
        return y

    def left_func(x, y, t):
        return -y

    def up_func(x, y, t):
        return -x

    def down_func(x, y, t):
        return x

    def diag_func(x, y, t):
        return y - x

    def circular_func(x, y, t):
        return -1 / 2 * (y ** 2 - y + x ** 2 - x)

    def swirl_func(x, y, t):
        T = 1.0
        return (
            1
            / np.pi
            * np.sin(np.pi * x / args.Lx) ** 2
            * np.sin(np.pi * y / args.Ly) ** 2
            * np.cos(np.pi * t / T)
        )

    if args.advection_function == "zero" or args.advection_function == "zeros":
        phi = phi_func(zeros_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "right":
        phi = phi_func(right_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "left":
        phi = phi_func(left_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "up":
        phi = phi_func(up_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "down":
        phi = phi_func(down_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "diagonal" or args.advection_function == "diag":
        phi = phi_func(diag_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "circular":
        phi = phi_func(circular_func, 0.0)
        return lambda zeta, t: phi
    elif args.advection_function == "swirl":
        return lambda zeta, t: phi_func(swirl_func, t)
    else:
        raise NotImplementedError
