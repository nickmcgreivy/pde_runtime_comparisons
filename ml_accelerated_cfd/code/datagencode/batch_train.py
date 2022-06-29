from jax import config
import time as time

config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import numpy as onp
from jax import value_and_grad, vmap
import sys
import h5py

from simulator import simulate_2D
from flux import Flux
from poissonbracket import get_poisson_bracket
from poissonsolver import get_poisson_solver
from rungekutta import FUNCTION_MAP
from training import (
    save_training,
    load_training,
    get_model,
    create_train_state,
    get_f_phi,
)
from diffusion import get_diffusion_func
from arguments import get_args
from helper import legendre_inner_product, inner_prod_with_legendre

PI = np.pi

def get_idx_gen(key, args, n_data):
    possible_idxs = np.arange(n_data)
    shuffle_idxs = jax.random.permutation(key, possible_idxs)

    counter = 0
    while counter + args.batch_size <= n_data:
        yield np.sort(shuffle_idxs[counter : counter + args.batch_size])
        counter += args.batch_size


def batch_gen(args, dset_a, t_arr, keys_arr):

    def gen(key, idxs):
        idxs = onp.array(idxs)
        a0 = dset_a[idxs, 0]
        t0 = t_arr[idxs, 0]
        key = keys_arr[idxs, :]
        a_data = dset_a[idxs, 1:]
        return {"a0": a0, "t0": t0, "a_data": a_data, "key": key}

    return gen

def create_f_poisson_solve_dict(args):
    dictionary = {}
    for order in args.orders:
        dictionary[order] = {}
        for up in args.upsampling:
            nx = args.nx_max // up
            ny = args.ny_max // up
            dictionary[order][up] = get_poisson_solver(
                args.poisson_dir, nx, ny, args.Lx, args.Ly, order
            )
    return dictionary


def batch_train_model(args, data_dir, save_dir):
    key = jax.random.PRNGKey(args.random_seed)


    if args.equation == "advection":
        f_poisson_solve_dict = None
    else:
        f_poisson_solve_dict = create_f_poisson_solve_dict(args)




    for i, order in enumerate(args.orders):
        f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, Flux.LEARNED)

        if args.diffusion_coefficient > 0.0:
            f_diffusion = get_diffusion_func(order, args.Lx, args.Ly, args.diffusion_coefficient)
        else:
            f_diffusion = None

        for j, up in enumerate(args.upsampling):

            model = get_model(args, order)

            # LOAD DATA
            f = h5py.File(
                "{}/{}_up{}_order{}.hdf5".format(data_dir, args.unique_id, up, order), "r"
            )
            dset_a = f["a_data"]
            t_arr = f["t_data"]
            keys_arr = f["key_data"]

            n_data = dset_a.shape[0]

            assert args.batch_size <= n_data

            num_epochs = int(
                np.ceil(args.training_iterations / n_data * args.batch_size)
            )

            nx = args.nx_max // up
            ny = args.ny_max // up
            dx = args.Lx / (nx)
            dy = args.Ly / (ny)
            dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)

            get_batch = batch_gen(args, dset_a, t_arr, keys_arr)


            if args.is_forcing:
                if args.equation == 'hw':
                    return NotImplementedError
                else:
                    leg_ip = np.asarray(legendre_inner_product(order))
                    ff = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
                    y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 2 * order + 1)
                    dx = args.Lx / nx
                    dy = args.Ly / ny
                    f_forcing = lambda zeta: (y_term - dx * dy * args.damping_coefficient * zeta * leg_ip) * args.forcing_coefficient
            else:
                f_forcing = None


            def loss_fn(params, batch):
                def get_loss(a0, t0, a_data, key):
                    key1, key2 = jax.random.split(key)
                    if args.equation == "advection":
                        f_phi = get_f_phi(key2, args, nx, ny, order)
                    else:
                        f_phi = lambda xi, t: f_poisson_solve_dict[order][up](xi)

                    return simulate_2D(
                        a0,
                        t0,
                        nx,
                        ny,
                        args.Lx,
                        args.Ly,
                        order,
                        dt,
                        args.num_unroll,
                        Flux.LEARNED,
                        alpha=args.alpha,
                        kappa=args.kappa,
                        equation=args.equation,
                        a_data=a_data,
                        f_poisson_bracket=f_poisson_bracket,
                        f_phi=f_phi,
                        f_diffusion=f_diffusion,
                        f_forcing=f_forcing,
                        rk=FUNCTION_MAP[args.runge_kutta],
                        model=model,
                        params=params,
                    )

                return np.mean(
                    vmap(get_loss)(
                        batch["a0"], batch["t0"], batch["a_data"], batch["key"]
                    )
                )

            grad_fn = jax.value_and_grad(loss_fn)

            @jax.jit
            def train_step(state, batch):
                loss, grads = grad_fn(state.params, batch)
                state = state.apply_gradients(grads=grads)
                return loss, state

            state = create_train_state(key, args, model, order)

            losses = []
            for e, epoch in enumerate(range(num_epochs)):
                key, subkey = jax.random.split(key)
                idx_gen = get_idx_gen(subkey, args, n_data)
                for i, idxs in enumerate(idx_gen):
                    key, subkey = jax.random.split(key, 2)
                    batch = get_batch(subkey, idxs)
                    loss, state = train_step(state, batch)
                    losses.append(loss)

            # write output
            save_training(state.params, losses, save_dir, args.unique_id, order, up)
