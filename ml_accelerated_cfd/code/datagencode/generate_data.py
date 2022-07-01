import jax.numpy as np
import jax
from jax import jit, vmap
import numpy as onp
import h5py
from functools import partial, reduce
from jax import config
from time import time

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

from basisfunctions import num_elements
from simulator import simulate_2D
from rungekutta import FUNCTION_MAP
from training import load_training, get_f_phi, get_initial_condition
from flux import Flux
from helper import f_to_DG, _evalf_2D_integrate, nabla
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from helper import legendre_inner_product, inner_prod_with_legendre
from baseline import vorticity, get_step_func, simulate_baseline, get_velocity, get_forcing
from plot_data import plot_DG_basis
import matplotlib.pyplot as plt


PI = np.pi

def create_dataset(args, N, data_dir, unique_id, up, order, nx, ny, nt):
    f = h5py.File(
        "{}/{}_up{}_order{}.hdf5".format(data_dir, unique_id, up, order),
        "w",
    )

    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        dset_a_new = f.create_dataset(
            "a_data", (N, nt, 2, nx, ny, num_elements(order)), dtype="float64"
        )
    else:
        dset_a_new = f.create_dataset(
            "a_data", (N, nt, nx, ny, num_elements(order)), dtype="float64"
        )
    dset_t_new = f.create_dataset("t_data", (N, nt), dtype="float64")
    dset_key_new = f.create_dataset("key_data", (N, 2), dtype="uint32")
    f.close()


def write_dataset(args, n, data_dir, unique_id, up, order, a_data, t_data, key):
    f = h5py.File(
        "{}/{}_up{}_order{}.hdf5".format(data_dir, unique_id, up, order),
        "r+",
    )
    dset_a = f["a_data"]
    dset_t = f["t_data"]
    dset_keys = f["key_data"]
    dset_a[n] = a_data
    dset_t[n] = t_data
    dset_keys[n] = key
    f.close()


@partial(
    jit,
    static_argnums=(
        1,
        2,
        3,
        4,
        7,
    ),
)
def convert_DG_representation(
    a, order_new, order_high, nx_new, ny_new, Lx, Ly, equation, n = 8
):
    """
    Inputs:
    a: (nt, nx, ny, num_elements(order_high))

    Outputs:
    a_converted: (nt, nx_new, ny_new, num_elements(order_new))
    """
    _, nx_high, ny_high = a.shape[0:3]
    dx_high = Lx / nx_high
    dy_high = Ly / ny_high

    def convert_repr(a):
        def f_high(x, y, t):
            return _evalf_2D_integrate(x, y, a, dx_high, dy_high, order_high)

        t0 = 0.0
        return f_to_DG(nx_new, ny_new, Lx, Ly, order_new, f_high, t0, n=n)

    vmap_convert_repr = vmap(convert_repr)
    if equation == "hw" or equation == "hasegawa_wakatani":
        zeta_cv = vmap_convert_repr(a[:, 0])
        n_cv = vmap_convert_repr(a[:, 1])
        return np.concatenate((zeta_cv[:, None], n_cv[:, None]), axis=1)
    else:
        return vmap_convert_repr(a)



####################
# GENERATE TRAIN DATA
####################


def generate_train_data(args, data_dir, N, T, seed):
    key = jax.random.PRNGKey(seed)

    t0 = 0.0
    dx = args.Lx / args.nx_max
    dy = args.Ly / args.ny_max

    t_downsample = [order * 2 + 1 for order in args.orders]
    prod_t_downsample = reduce(lambda x, y: x * y, t_downsample, 1)
    while prod_t_downsample < 2 * args.order_max + 1:
        prod_t_downsample *= 2
    time_upsampling = [prod_t_downsample // t for t in t_downsample]
    dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / prod_t_downsample
    dt_burn_in = dt

    nt_burn_in = int(args.burn_in_time // dt_burn_in)
    nt_sim = int(T / dt / args.chunks_per_sim)


    if args.equation == "advection":
        f_poisson_solve = None
    else:
        f_poisson_solve = get_poisson_solver(
            args.poisson_dir, args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max
        )
    f_poisson_bracket_exact = get_poisson_bracket(args.poisson_dir, args.order_max, args.exact_flux)

    if args.diffusion_coefficient > 0.0:
        f_diffusion_exact = get_diffusion_func(args.order_max, args.Lx, args.Ly, args.diffusion_coefficient)
    else:
        f_diffusion_exact = None

    if args.is_forcing:
        if args.equation == 'hw':
            return NotImplementedError
        else:
            leg_ip_exact = np.asarray(legendre_inner_product(args.order_max))
            ffe = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
            y_term_exact = inner_prod_with_legendre(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, ffe, 0.0, n = 2 * args.order_max + 1)
            dx = args.Lx / args.nx_max
            dy = args.Ly / args.ny_max
            f_forcing_exact = lambda zeta: (y_term_exact - dx * dy * args.damping_coefficient * zeta * leg_ip_exact) * args.forcing_coefficient
    else:
        f_forcing_exact = None 


    @jit
    def burn_in(key):
        dx = args.Lx / args.nx_max
        dy = args.Ly / args.ny_max
        key1, key2 = jax.random.split(key)
        t0 = 0.0
        f_init = get_initial_condition(key1, args)

        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            n0 = f_to_DG(
                args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8
            )
            f_phi0 = lambda x, y, t: np.sum(
                -1
                * nabla(lambda x, y: f_init(x, y, t0))(np.asarray([x]), np.asarray([y]))
            )
            zeta0 = f_to_DG(
                args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_phi0, t0, n = 8
            )
            a0 = np.concatenate((zeta0[None], n0[None]))
        else:
            a0 = f_to_DG(
                args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8
            )

        if args.equation == "advection":
            f_phi = get_f_phi(key2, args, args.nx_max, args.ny_max, args.order_max)
        else:
            f_phi = lambda xi, t: f_poisson_solve(xi)

        return simulate_2D(
            a0,
            t0,
            args.nx_max,
            args.ny_max,
            args.Lx,
            args.Ly,
            args.order_max,
            dt_burn_in,
            nt_burn_in,
            args.exact_flux,
            alpha=args.alpha,
            kappa=args.kappa,
            equation=args.equation,
            output=False,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
        )

    @partial(
        jit,
        static_argnums=(
            3,
        ),
    )
    def simulate(key, a_in, t_in, upsampling):

        _, key2 = jax.random.split(key)

        if args.equation == "advection":
            f_phi = get_f_phi(key2, args, args.nx_max, args.ny_max, args.order_max)
        else:
            f_phi = lambda xi, t: f_poisson_solve(xi)

        a_data = simulate_2D(
            a_in,
            t_in,
            args.nx_max,
            args.ny_max,
            args.Lx,
            args.Ly,
            args.order_max,
            dt,
            nt_sim // upsampling,
            args.exact_flux,
            alpha=args.alpha,
            kappa=args.kappa,
            equation=args.equation,
            a_data=None,
            output=True,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
            inner_loop_steps=upsampling,
        )

        a_data = np.concatenate((a_in[None], a_data))
        t_data = t_in + np.arange(nt_sim // upsampling + 1) * (dt * upsampling)
        return {"a": a_data, "t": t_data}

    for i, order in enumerate(args.orders):
        for j, up in enumerate(args.upsampling):
            t_up = time_upsampling[i] * up
            nx = args.nx_max // up
            ny = args.ny_max // up

            create_dataset(
                args, N * args.chunks_per_sim, data_dir, args.unique_id, up, order, nx, ny, args.num_unroll + 1
            )

            assert ((nt_sim) // t_up) + 1 >= args.num_unroll + 1



    for n in range(N):
        key, subkey = jax.random.split(key)

        t1 = time()

        a_f, t_f = burn_in(subkey)

        for j in range(args.chunks_per_sim):
            for i, order in enumerate(args.orders):
                for up in args.upsampling:

                    t_up = time_upsampling[i] * up

                    data = simulate(subkey, a_f, t_f, t_up)

                    a_converted = convert_DG_representation(
                        data["a"],
                        order,
                        args.order_max,
                        args.nx_max // up,
                        args.ny_max // up,
                        args.Lx,
                        args.Ly,
                        args.equation,
                    )
                    t_converted = data["t"]

                    n_max = a_converted.shape[0]
                    assert n_max >= (args.num_unroll + 1)
                    start = jax.random.randint(key, (1,), minval=0, maxval=n_max - (args.num_unroll + 1))[0] 
                    end = start + args.num_unroll + 1
                    a_chunk = a_converted[start:end]
                    t_chunk = t_converted[start:end]

                    key, _ = jax.random.split(key)


                    write_dataset(
                        args,
                        n * args.chunks_per_sim + j, ############
                        data_dir,
                        args.unique_id,
                        up,
                        order,
                        a_chunk,
                        t_chunk,
                        subkey,
                    )

            a_f = data["a"][-1]
            t_f = data["t"][-1]

        t2 = time()
        print(t2 - t1)









####################
# GENERATE TEST DATA
####################


def generate_test_data(args, data_dir, N, T, seed, inner_loop_steps):
    key = jax.random.PRNGKey(seed)

    t0 = 0.0
    dx = args.Lx / args.nx_max
    dy = args.Ly / args.ny_max

    t_downsample = [order * 2 + 1 for order in args.orders]
    prod_t_downsample = reduce(lambda x, y: x * y, t_downsample, 1)
    while prod_t_downsample < 2 * args.order_max + 1:
        prod_t_downsample *= 2
    time_upsampling = [prod_t_downsample // t for t in t_downsample]
    dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / prod_t_downsample
    dt_burn_in = dt
    nt = int(T / dt / inner_loop_steps)
    nt_burn_in = int(args.burn_in_time // dt_burn_in)
    if args.equation == "advection":
        f_poisson_solve = None
    else:
        f_poisson_solve = get_poisson_solver(
            args.poisson_dir, args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max
        )
    f_poisson_bracket_exact = get_poisson_bracket(args.poisson_dir, args.order_max, args.exact_flux)

    if args.diffusion_coefficient > 0.0:
        f_diffusion_exact = get_diffusion_func(args.order_max, args.Lx, args.Ly, args.diffusion_coefficient)
    else:
        f_diffusion_exact = None

    if args.is_forcing:
        if args.equation == 'hw':
            return NotImplementedError
        else:
            leg_ip_exact = np.asarray(legendre_inner_product(args.order_max))
            ffe = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
            y_term_exact = inner_prod_with_legendre(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, ffe, 0.0, n = 2 * args.order_max + 1)
            dx = args.Lx / args.nx_max
            dy = args.Ly / args.ny_max
            f_forcing_exact = lambda zeta: (y_term_exact - dx * dy * args.damping_coefficient * zeta * leg_ip_exact) * args.forcing_coefficient
    else:
        f_forcing_exact = None 


    @jit
    def simulate(key):
        dx = args.Lx / args.nx_max
        dy = args.Ly / args.ny_max
        key1, key2 = jax.random.split(key)
        t0 = 0.0
        f_init = get_initial_condition(key1, args)

        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            n0 = f_to_DG(
                args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8
            )
            f_phi0 = lambda x, y, t: np.sum(
                -1
                * nabla(lambda x, y: f_init(x, y, t0))(np.asarray([x]), np.asarray([y]))
            )
            zeta0 = f_to_DG(
                args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_phi0, t0, n = 8
            )
            a0 = np.concatenate((zeta0[None], n0[None]))
        else:
            a0 = f_to_DG(
                args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8
            )

        if args.equation == "advection":
            f_phi = get_f_phi(key2, args, args.nx_max, args.ny_max, args.order_max)
        else:
            f_phi = lambda xi, t: f_poisson_solve(xi)

        a_burn_in, t_burn_in = simulate_2D(
            a0,
            t0,
            args.nx_max,
            args.ny_max,
            args.Lx,
            args.Ly,
            args.order_max,
            dt_burn_in,
            nt_burn_in,
            args.exact_flux,
            alpha=args.alpha,
            kappa=args.kappa,
            equation=args.equation,
            output=False,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
        )

        a_data = simulate_2D(
            a_burn_in,
            t_burn_in,
            args.nx_max,
            args.ny_max,
            args.Lx,
            args.Ly,
            args.order_max,
            dt,
            nt-1,
            args.exact_flux,
            alpha=args.alpha,
            kappa=args.kappa,
            equation=args.equation,
            a_data=None,
            output=True,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
            inner_loop_steps=inner_loop_steps,
        )

        a_data = np.concatenate((a_burn_in[None], a_data))
        t_data = t_burn_in + np.arange(nt) * dt
        return {"a": a_data, "t": t_data}

    for i, order in enumerate(args.orders):
        for j, up in enumerate(args.upsampling):
            t_up = time_upsampling[i] * up
            nx = args.nx_max // up
            ny = args.ny_max // up

            create_dataset(
                args, N, data_dir, args.unique_id, up, order, nx, ny, int(np.ceil(nt / t_up))
            )

    for n in range(N):
        key, subkey = jax.random.split(key)
        t1 = time()
        data = simulate(subkey)
        t2 = time()
        print(t2 - t1)

        for i, order in enumerate(args.orders):
            for up in args.upsampling:
                t_up = time_upsampling[i] * up
                
                a_converted = convert_DG_representation(
                    data["a"][::t_up],
                    order,
                    args.order_max,
                    args.nx_max // up,
                    args.ny_max // up,
                    args.Lx,
                    args.Ly,
                    args.equation,
                )
                t_converted = data["t"][::t_up]

                write_dataset(
                    args,
                    n,
                    data_dir,
                    args.unique_id,
                    up,
                    order,
                    a_converted,
                    t_converted,
                    subkey,
                )











####################
# GENERATE EVAL DATA
####################




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


def create_f_diffusion_dict(args):
    dictionary = {}
    for order in args.orders:
        dictionary[order] = get_diffusion_func(order, args.Lx, args.Ly, args.diffusion_coefficient)
    return dictionary

def generate_eval_data(args, data_dir, N, T, flux, unique_id, seed, inner_loop_steps):
    key = jax.random.PRNGKey(seed)

    f_poisson_bracket_dict = create_f_poisson_bracket_dict(args, flux)
    f_poisson_bracket_exact = get_poisson_bracket(args.poisson_dir, args.order_max, args.exact_flux)
    if args.equation == "advection":
        f_poisson_solve_dict = None
    else:
        f_poisson_solve_dict = create_f_poisson_solve_dict(args)
        f_poisson_exact = get_poisson_solver(
            args.poisson_dir, args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max
        )

    if args.diffusion_coefficient > 0.0:
        f_diffusion_exact = get_diffusion_func(args.order_max, args.Lx, args.Ly, args.diffusion_coefficient)
        f_diffusion_dict = create_f_diffusion_dict(args)
    else:
        f_diffusion_exact = None


    if args.is_forcing:
        if args.equation == 'hw':
            return NotImplementedError
        else:
            leg_ip_exact = np.asarray(legendre_inner_product(args.order_max))
            ffe = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
            y_term_exact = inner_prod_with_legendre(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, ffe, 0.0, n = 2 * args.order_max + 1)
            dx = args.Lx / args.nx_max
            dy = args.Ly / args.ny_max
            f_forcing_exact = lambda zeta: (y_term_exact - dx * dy * args.damping_coefficient * zeta * leg_ip_exact) * args.forcing_coefficient
    else:
        f_forcing_exact = None 

    @partial(
        jit,
        static_argnums=(
            1,
            2,
        ),
    )
    def simulate(key, up, order, params=None):
        key1, key2 = jax.random.split(key)
        f_init = get_initial_condition(key1, args)
        nx = args.nx_max // up
        ny = args.ny_max // up
        dx = args.Lx / (nx)
        dy = args.Ly / (ny)
        dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
        t0 = 0.0
        nt = int(T / dt / inner_loop_steps)

        dx_min = args.Lx / (args.nx_max)
        dy_min = args.Ly / (args.ny_max)
        dt_burn_in = args.cfl_safety * ((dx_min * dy_min) / (dx_min + dy_min)) / (2 * args.order_max + 1)
        nt_burn_in = int(args.burn_in_time // dt_burn_in)

        if flux == Flux.LEARNED:
            model, _, _ = load_training(args, args.param_dir, args.unique_id, order, up)
        else:
            model = None

        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            n0 = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8)
            f_phi0 = lambda x, y, t: np.sum(
                -1
                * nabla(lambda x, y: f_init(x, y, t0))(np.asarray([x]), np.asarray([y]))
            )
            zeta0 = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_phi0, t0, n = 8)
            a0 = np.concatenate((zeta0[None], n0[None]))
        else:
            a0 = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8)

        f_poisson_bracket = f_poisson_bracket_dict[order]
        if args.equation == "advection":
            f_phi = get_f_phi(key2, args, nx, ny, order)
            f_phi_exact = get_f_phi(key2, args, args.nx_max, args.ny_max, args.order_max)
        else:
            f_poisson_solve = f_poisson_solve_dict[order][up]
            f_phi = lambda zeta, t: f_poisson_solve(zeta)
            f_phi_exact = lambda zeta, t: f_poisson_exact(zeta)

        if args.diffusion_coefficient > 0.0:
            f_diffusion = f_diffusion_dict[order]
        else:
            f_diffusion = None

        if args.is_forcing:
            leg_ip = np.asarray(legendre_inner_product(order))
            ff = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
            y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 2 * order + 1)
            dx = args.Lx / nx
            dy = args.Ly / ny
            f_forcing_sim = lambda zeta: (y_term - dx * dy * args.damping_coefficient * zeta * leg_ip) * args.forcing_coefficient
        else:
            f_forcing_sim = None

        a_burn_in, t_burn_in = simulate_2D(
            a0,
            t0,
            args.nx_max,
            args.ny_max,
            args.Lx,
            args.Ly,
            args.order_max,
            dt_burn_in,
            nt_burn_in,
            args.exact_flux,
            alpha=args.alpha,
            kappa=args.kappa,
            equation=args.equation,
            output=False,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi_exact,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
        )

        a_burn_in = convert_DG_representation(
            a_burn_in[None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
        )[0]

        a_data = simulate_2D(
            a_burn_in,
            t_burn_in,
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
            a_data=None,
            output=True,
            f_phi=f_phi,
            f_poisson_bracket=f_poisson_bracket,
            f_diffusion=f_diffusion,
            f_forcing=f_forcing_sim,
            rk=FUNCTION_MAP[args.runge_kutta],
            inner_loop_steps=inner_loop_steps,
        )
        a_data = np.concatenate((a_burn_in[None], a_data))
        t_data = t_burn_in + np.arange(nt) * dt
        return {"a": a_data, "t": t_data}

    for order in args.orders:
        for up in args.upsampling:
            nx = args.nx_max // up
            ny = args.ny_max // up
            dx = args.Lx / nx
            dy = args.Ly / ny
            dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
            nt = int(T / dt / inner_loop_steps)
            create_dataset(args, N, data_dir, unique_id, up, order, nx, ny, nt)

    for n in range(N):
        key, subkey = jax.random.split(key)
        for order in args.orders:
            for up in args.upsampling:
                if flux == Flux.LEARNED:
                    _, params, _ = load_training(
                        args, args.param_dir, args.unique_id, order, up
                    )
                else:
                    _, params = None, None
                t1 = time()
                data = simulate(subkey, up, order, params=params)
                t2 = time()
                print(t2 - t1)

                write_dataset(
                    args,
                    n,
                    data_dir,
                    unique_id,
                    up,
                    order,
                    data["a"],
                    data["t"],
                    subkey,
                )



####################
# TEST RUNTIME
####################


def print_runtime(args, T, flux, unique_id, seed, inner_loop_steps, nx, ny, order):
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    f_init = get_initial_condition(key1, args)

    dx = args.Lx / (nx)
    dy = args.Ly / (ny)
    dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
    t0 = 0.0
    nt = int(T / dt / inner_loop_steps)


    f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)


    if args.equation == "advection":
        f_phi = get_f_phi(key2, args, nx, ny, order)
    else:
        f_poisson_solve = get_poisson_solver(
            args.poisson_dir, nx, ny, args.Lx, args.Ly, order
        )
        f_phi = lambda zeta, t: f_poisson_solve(zeta)


    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        n0 = f_to_DG(nx, ny, args.Lx, args.Ly, order, f_init, t0, n = 8)
        f_phi0 = lambda x, y, t: np.sum(
            -1
            * nabla(lambda x, y: f_init(x, y, t0))(np.asarray([x]), np.asarray([y]))
        )
        zeta0 = f_to_DG(nx, ny, args.Lx, args.Ly, order, f_phi0, t0, n = 8)
        a0 = np.concatenate((zeta0[None], n0[None]))
    else:
        a0 = f_to_DG(nx, ny, args.Lx, args.Ly, order, f_init, t0, n = 8)

       
    if flux == Flux.LEARNED:
        raise NotImplementedError
    else:
        model = None 


    if args.diffusion_coefficient > 0.0:
        f_diffusion = get_diffusion_func(order, args.Lx, args.Ly, args.diffusion_coefficient)
    else:
        f_diffusion = None

    if args.is_forcing:
        leg_ip = np.asarray(legendre_inner_product(order))
        ff = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
        y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 2 * order + 1)
        dx = args.Lx / nx
        dy = args.Ly / ny
        f_forcing_sim = lambda zeta: (y_term - dx * dy * args.damping_coefficient * zeta * leg_ip) * args.forcing_coefficient
    else:
        f_forcing_sim = None

    @jit
    def simulate(params=None):
        model = None

        return simulate_2D(
            a0,
            t0,
            nx,
            ny,
            args.Lx,
            args.Ly,
            order,
            dt,
            nt,
            flux,
            alpha=args.alpha,
            kappa=args.kappa,
            model=model,
            params=params,
            equation=args.equation,
            a_data=None,
            output=False,
            f_phi=f_phi,
            f_poisson_bracket=f_poisson_bracket,
            f_diffusion=f_diffusion,
            f_forcing=f_forcing_sim,
            rk=FUNCTION_MAP[args.runge_kutta],
            inner_loop_steps=inner_loop_steps,
        )



    if flux == Flux.LEARNED:
        raise NotImplementedError
    else:
        _, params = None, None

    print("nx, ny are {}, {}".format(nx, ny))
    print("order is {}".format(order))
    print("first compilation")
    t1 = time()
    data = simulate(params = params)
    t2 = time()
    print("runtime for T = {} is {}".format(T, t2 - t1))

    print("second compilation")
    t1 = time()
    data = simulate(params = params)
    t2 = time()
    print("runtime for T = {} is {}".format(T, t2 - t1))




####################
# COMPUTE CORRCOEF
####################

def vorticity_to_velocity(args, a, f_poisson):
    H = f_poisson(a)
    nx, ny, _ = H.shape
    dx = args.Lx / nx
    dy = args.Ly / ny

    u_y = -(H[:,:,2] - H[:,:,3]) / dx
    u_x = (H[:,:,2] - H[:,:,1]) / dy
    return u_x, u_y

def shift_down_left(a):
    return (a + np.roll(a, 1, axis=1) + np.roll(a, 1, axis=0) + np.roll(np.roll(a, 1, axis=0), 1, axis=1)) / 4


def compute_corrcoef(args, orders, nxs, nxs_baseline, baseline_dt_reductions, Tf, Np):


    key = jax.random.PRNGKey(args.random_seed)
    key1, key2 = jax.random.split(key)
    t0 = 0.0

    ###### init vorticity
    f_init = get_initial_condition(key1, args)
    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        raise NotImplementedError
    else:
        a0 = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8)




    #########
    # burn_in
    #########
    dx_min = args.Lx / (args.nx_max)
    dy_min = args.Ly / (args.ny_max)
    dt_exact = args.cfl_safety * ((dx_min * dy_min) / (dx_min + dy_min)) / (2 * args.order_max + 1)
    nt_burn_in = int(args.burn_in_time // dt_exact) + 1
    dt_burn_in = args.burn_in_time / nt_burn_in
    

    f_poisson_exact = get_poisson_solver(
        args.poisson_dir, args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max
    )
    f_phi_exact = lambda zeta, t: f_poisson_exact(zeta)

    if args.diffusion_coefficient > 0.0:
        f_diffusion_exact = get_diffusion_func(args.order_max, args.Lx, args.Ly, args.diffusion_coefficient)
    else:
        f_diffusion_exact = None

    if args.is_forcing:
        if args.equation == 'hw':
            return NotImplementedError
        else:
            leg_ip_exact = np.asarray(legendre_inner_product(args.order_max))
            ffe = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
            y_term_exact = inner_prod_with_legendre(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, ffe, 0.0, n = 8)
            dx_min = args.Lx / args.nx_max
            dy_min = args.Ly / args.ny_max
            f_forcing_exact = lambda zeta: (y_term_exact - dx_min * dy_min * args.damping_coefficient * zeta * leg_ip_exact) * args.forcing_coefficient
    else:
        f_forcing_exact = None

    f_poisson_bracket_exact = get_poisson_bracket(args.poisson_dir, args.order_max, args.exact_flux)

    @partial(
        jit,
        static_argnums=(
            2
        ),
    )
    def sim_exact(a_i, t_i, nt, dt):
        a_f, t_f = simulate_2D(
            a_i,
            t_i,
            args.nx_max,
            args.ny_max,
            args.Lx,
            args.Ly,
            args.order_max,
            dt,
            nt,
            args.exact_flux,
            alpha=args.alpha,
            kappa=args.kappa,
            equation=args.equation,
            output=False,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi_exact,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
            inner_loop_steps=1,
        )
        return a_f, t_f

    if args.burn_in_time > 0.0:
        a_burn_in_exact, t_burn_in = sim_exact(a0, t0, nt_burn_in, dt_burn_in)
    else:
        a_burn_in_exact, t_burn_in = a0, t0



    ########
    # Store exact data at Np intervals 
    ########

    T_chunk = Tf / Np

    assert Np >= 1

    nt_sim_exact = int(T_chunk // dt_exact) + 1
    dt_exact = T_chunk / nt_sim_exact
    a_exact_all = a_burn_in_exact[...,None]
    a_exact = a_burn_in_exact
    t_exact = t_burn_in

    for j in range(Np):
        a_exact, t_exact = sim_exact(a_exact, t_exact, nt_sim_exact, dt_exact)
        a_exact_all = np.concatenate((a_exact_all, a_exact[...,None]),axis=-1)


    def print_correlation(a_f, order, nx, ny, j):
        a_e = convert_DG_representation(a_exact_all[...,j][None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation)[0]
        M = np.concatenate([a_f[:,:,0].reshape(-1)[:,None], a_e[:,:,0].reshape(-1)[:,None]],axis=1)
        print("Correlation coefficient for T = {:.1f}, order = {}, nx & ny = {} is {}".format(j * T_chunk, order, nx, np.corrcoef(M.T)[0,1]))


    def downsample_ux(u_x, F):
        nx, ny = u_x.shape
        assert nx % F == 0
        return np.mean(u_x[F-1::F,:].reshape(nx // F, ny // F, F), axis=2)
        

    def downsample_uy(u_y, F):
        nx, ny = u_y.shape
        assert ny % F == 0
        return np.mean(u_y[:, F-1::F].reshape(nx // F, F, ny // F), axis=1)

    ######
    # start with baseline implementation
    ######
    for nx in nxs_baseline:
        for BASELINE_DT_REDUCTION in baseline_dt_reductions:
            print("baseline, nx is {}, reduction is {}".format(nx, BASELINE_DT_REDUCTION))
            
            ny = nx
            dx = args.Lx / (nx)
            dy = args.Ly / (ny)
            dt = (args.cfl_safety / BASELINE_DT_REDUCTION) * ((dx * dy) / (dx + dy))
            nt_chunk = int(T_chunk // dt) + 1
            dt = T_chunk / nt_chunk
            if args.is_forcing:
                f_forcing_baseline = get_forcing(args, nx, ny)
            else:
                f_forcing_baseline = None
            step_func = get_step_func(args.diffusion_coefficient, dt, forcing=f_forcing_baseline)

            a_burn_in = convert_DG_representation(a_burn_in_exact[None], 0, args.order_max, nxs_baseline[-1], nxs_baseline[-1], args.Lx, args.Ly, args.equation)[0]
            f_ps_exact = jit(get_poisson_solver(args.poisson_dir, nxs_baseline[-1], nxs_baseline[-1], args.Lx, args.Ly, 0))
            u_x_burn_in, u_y_burn_in = vorticity_to_velocity(args, a_burn_in, f_ps_exact)


            DS_FAC = nxs_baseline[-1] // nx
            u_x_burn_in = downsample_ux(u_x_burn_in, DS_FAC)
            u_y_burn_in = downsample_uy(u_y_burn_in, DS_FAC)
            v_burn_in = get_velocity(args, u_x_burn_in, u_y_burn_in)
            v_f = v_burn_in

            """
            PLOTMAX = 2.5 * args.amplitude_max
            a_f = shift_down_left(vorticity(v_f))[..., None]
            plot_DG_basis(nx, ny, args.Lx, args.Ly, 0, a_f, title="baseline, t = {}".format(0), plotting_density=1, vmax=PLOTMAX)
            a_e = convert_DG_representation(a_exact_all[...,0][None], 0, args.order_max, nx, ny, args.Lx, args.Ly, args.equation)[0]
            plot_DG_basis(nx, ny, args.Lx, args.Ly, 0, a_e, title="exact, t = {}".format(0), plotting_density=1, vmax=PLOTMAX)
            """
            
            for j in range(Np+1):
                a_f = shift_down_left(vorticity(v_f))[..., None]
                print_correlation(a_f, 0, nx, ny, j)
                v_f = simulate_baseline(v_f, step_func, nt_chunk)
            
            """
            a_f = shift_down_left(vorticity(v_f))[..., None]
            plot_DG_basis(nx, ny, args.Lx, args.Ly, 0, a_f, title="baseline, t = {}".format(Np), plotting_density=1, vmax=PLOTMAX)
            a_e = convert_DG_representation(a_exact_all[...,Np][None], 0, args.order_max, nx, ny, args.Lx, args.Ly, args.equation)[0]
            plot_DG_basis(nx, ny, args.Lx, args.Ly, 0, a_e, title="exact, t = {}".format(Np), plotting_density=1, vmax=PLOTMAX)
            plt.show()
            """
    

    ###### 
    # loop through orders and nxs
    ######

    for o, order in enumerate(orders):
        if order == 0:
            flux = Flux.VANLEER
        else:
            flux = Flux.UPWIND

        for nx in nxs[o]:
            print("nx is {}".format(nx))
            ny = nx
            dx = args.Lx / (nx)
            dy = args.Ly / (ny)
            dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
            nt_chunk = int(T_chunk // dt) + 1
            dt = T_chunk / nt_chunk

            f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)

            if args.equation == "advection":
                f_phi = get_f_phi(key2, args, nx, ny, order)
            else:
                f_poisson_solve = get_poisson_solver(
                    args.poisson_dir, nx, ny, args.Lx, args.Ly, order
                )
                f_phi = lambda zeta, t: f_poisson_solve(zeta)
            
            model = None 


            if args.diffusion_coefficient > 0.0:
                f_diffusion = get_diffusion_func(order, args.Lx, args.Ly, args.diffusion_coefficient)
            else:
                f_diffusion = None

            if args.is_forcing:
                leg_ip = np.asarray(legendre_inner_product(order))
                ff = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
                y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 8)
                dx = args.Lx / nx
                dy = args.Ly / ny
                f_forcing_sim = lambda zeta: (y_term - dx * dy * args.damping_coefficient * zeta * leg_ip) * args.forcing_coefficient
            else:
                f_forcing_sim = None

            @partial(
                jit,
                static_argnums=(
                    2
                ),
            )
            def simulate(a_i, t_i, nt, dt, params=None):
                model = None
                
                return simulate_2D(
                    a_i,
                    t_i,
                    nx,
                    ny,
                    args.Lx,
                    args.Ly,
                    order,
                    dt,
                    nt,
                    flux,
                    alpha=args.alpha,
                    kappa=args.kappa,
                    model=model,
                    params=params,
                    equation=args.equation,
                    a_data=None,
                    output=False,
                    f_phi=f_phi,
                    f_poisson_bracket=f_poisson_bracket,
                    f_diffusion=f_diffusion,
                    f_forcing=f_forcing_sim,
                    rk=FUNCTION_MAP[args.runge_kutta],
                    inner_loop_steps=1,
                )


            a_burn_in = convert_DG_representation(
                a_burn_in_exact[None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
            )[0]
            a_f, t_f = a_burn_in, t_burn_in

            for j in range(Np+1):
                print_correlation(a_f, order, nx, ny, j)
                a_f, t_f = simulate(a_f, t_f, nt_chunk, dt)






#################################################
# COMPUTE RUNTIME
#################################################




def compute_runtime(args, orders, nxs, nxs_baseline, baseline_dt_reduction, Tf):


    key = jax.random.PRNGKey(args.random_seed)
    key1, key2 = jax.random.split(key)
    t0 = 0.0

    ###### init vorticity
    f_init = get_initial_condition(key1, args)
    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        raise NotImplementedError
    else:
        a0 = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8)


    def downsample_ux(u_x, F):
        nx, ny = u_x.shape
        assert nx % F == 0
        return np.mean(u_x[F-1::F,:].reshape(nx // F, ny // F, F), axis=2)
        

    def downsample_uy(u_y, F):
        nx, ny = u_y.shape
        assert ny % F == 0
        return np.mean(u_y[:, F-1::F].reshape(nx // F, F, ny // F), axis=1)


    ######
    # start with baseline implementation
    ######
    a_init = convert_DG_representation(a0[None], 0, args.order_max, nxs_baseline[-1], nxs_baseline[-1], args.Lx, args.Ly, args.equation)[0]
    f_ps_exact = jit(get_poisson_solver(args.poisson_dir, nxs_baseline[-1], nxs_baseline[-1], args.Lx, args.Ly, 0))
    u_x_init, u_y_init = vorticity_to_velocity(args, a_init, f_ps_exact)

    for nx in nxs_baseline:
        
        ny = nx
        dx = args.Lx / (nx)
        dy = args.Ly / (ny)
        dt = (args.cfl_safety / baseline_dt_reduction) * ((dx * dy) / (dx + dy))
        nt = int(Tf // dt) + 1
        dt = Tf / nt
        if args.is_forcing:
            f_forcing_baseline = get_forcing(args, nx, ny)
        else:
            f_forcing_baseline = None
        step_func = get_step_func(args.diffusion_coefficient, dt, forcing=f_forcing_baseline)


        DS_FAC = nxs_baseline[-1] // nx
        u_x_init_ds = downsample_ux(u_x_init, DS_FAC)
        u_y_init_ds = downsample_uy(u_y_init, DS_FAC)
        v_init = get_velocity(args, u_x_init_ds, u_y_init_ds)

        
        ### time 1
        t1 = time()
        _, _ = simulate_baseline(v_init, step_func, nt)
        t2 = time()
        _, _ = simulate_baseline(v_init, step_func, nt)
        t3 = time()
        print("baseline, Tf = {}, nx = {}, compile time = {:.5f}, second runtime = {:.5f}".format(Tf, nx, (t2 - t1), (t3 - t2)))


    ###### 
    # loop through orders and nxs
    ######

    for o, order in enumerate(orders):
        if order == 0:
            flux = Flux.VANLEER
        else:
            flux = Flux.UPWIND

        for nx in nxs[o]:
            print("nx is {}".format(nx))
            ny = nx
            dx = args.Lx / (nx)
            dy = args.Ly / (ny)
            dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
            nt = int(Tf // dt) + 1
            dt = Tf / nt
            f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
            if args.equation == "advection":
                f_phi = get_f_phi(key2, args, nx, ny, order)
            else:
                f_poisson_solve = get_poisson_solver(
                    args.poisson_dir, nx, ny, args.Lx, args.Ly, order
                )
                f_phi = lambda zeta, t: f_poisson_solve(zeta)

               
            if flux == Flux.LEARNED:
                raise NotImplementedError
            else:
                model = None 


            if args.diffusion_coefficient > 0.0:
                f_diffusion = get_diffusion_func(order, args.Lx, args.Ly, args.diffusion_coefficient)
            else:
                f_diffusion = None

            if args.is_forcing:
                leg_ip = np.asarray(legendre_inner_product(order))
                ff = lambda x, y, t: -4 * (2 * PI / args.Ly) * np.cos(4 * (2 * PI / args.Ly) * y)
                y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 8)
                dx = args.Lx / nx
                dy = args.Ly / ny
                f_forcing_sim = lambda zeta: (y_term - dx * dy * args.damping_coefficient * zeta * leg_ip) * args.forcing_coefficient
            else:
                f_forcing_sim = None

            @partial(
                jit,
                static_argnums=(
                    2
                ),
            )
            def simulate(a_i, t_i, nt, dt, params=None):
                model = None
                
                return simulate_2D(
                    a_i,
                    t_i,
                    nx,
                    ny,
                    args.Lx,
                    args.Ly,
                    order,
                    dt,
                    nt,
                    flux,
                    alpha=args.alpha,
                    kappa=args.kappa,
                    model=model,
                    params=params,
                    equation=args.equation,
                    a_data=None,
                    output=False,
                    f_phi=f_phi,
                    f_poisson_bracket=f_poisson_bracket,
                    f_diffusion=f_diffusion,
                    f_forcing=f_forcing_sim,
                    rk=FUNCTION_MAP[args.runge_kutta],
                    inner_loop_steps=1,
                )


            a_init = convert_DG_representation(
                a0[None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
            )[0]

            t1 = time()
            _, _ = simulate(a_init, t0, nt, dt)
            t2 = time()
            _, _ = simulate(a_init, t0, nt, dt)
            t3 = time()
            print("order = {}, Tf = {}, nx = {}, compile time = {:.5f}, second runtime = {:.5f}".format(order, Tf, nx, (t2 - t1), (t3 - t2)))






