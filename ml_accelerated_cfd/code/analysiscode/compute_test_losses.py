from jax import config
import time as time

config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import numpy as onp
from jax import value_and_grad, vmap
import sys
import h5py
from functools import partial

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
from helper import legendre_inner_product, inner_prod_with_legendre

from arguments import get_args


PI = np.pi

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


def create_f_poisson_bracket_dict(args, flux):
    dictionary = {}
    for order in args.orders:
        dictionary[order] = get_poisson_bracket(args.poisson_dir, order, flux)
    return dictionary


def compute_losses(args, data_dir, flux, save_id, inner_loop_steps, mean_loss=True):
    f_poisson_bracket_dict = create_f_poisson_bracket_dict(args, flux)
    if args.equation == "advection":
        f_poisson_solve_dict = None
    else:
        f_poisson_solve_dict = create_f_poisson_solve_dict(args)

    for i, order in enumerate(args.orders):
        if args.diffusion_coefficient > 0.0:
            f_diffusion = get_diffusion_func(order, args.Lx, args.Ly, args.diffusion_coefficient)
        else:
            f_diffusion = None

        f_poisson_bracket = f_poisson_bracket_dict[order]
        for j, up in enumerate(args.upsampling):
            f = h5py.File(
                "{}/{}_up{}_order{}.hdf5".format(data_dir, args.unique_id, up, order),
                "r",
            )

            dset_a = f["a_data"]
            dset_t = f["t_data"]
            dset_keys = f["key_data"]
            nruns, nt, _, _, _ = dset_a.shape
            nx = args.nx_max // up
            ny = args.ny_max // up
            dx = args.Lx / (nx)
            dy = args.Ly / (ny)
            dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)

            if flux == Flux.LEARNED:
                model, params, _ = load_training(
                    args, args.param_dir, args.unique_id, order, up
                )
            else:
                model, params = None, None
            

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

            @jax.jit
            def loss_fn(a0, t0, a_data, key):
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
                    nt-1,
                    flux,
                    alpha=args.alpha,
                    kappa=args.kappa,
                    model=model,
                    params=params,
                    equation=args.equation,
                    a_data=a_data,
                    f_phi=f_phi,
                    f_diffusion = f_diffusion,
                    f_forcing=f_forcing,
                    f_poisson_bracket=f_poisson_bracket,
                    rk=FUNCTION_MAP[args.runge_kutta],
                    square_root_loss=False,
                    mean_loss=mean_loss,
                    inner_loop_steps=inner_loop_steps,
                )

            losses = []
            for n in range(nruns):
                losses.append(
                    loss_fn(
                        dset_a[n, 0],
                        dset_t[n, 0],
                        dset_a[n, 1:],
                        dset_keys[n],
                    )
                )

            with open(
                "{}/{}_up{}_order{}_losses.txt".format(data_dir, save_id, up, order), "w"
            ) as f:
                for listitem in losses:
                    if mean_loss:
                        f.write("%s\n" % listitem)
                    else:
                        f.write("%s\n" % list(onp.asarray(listitem)))
