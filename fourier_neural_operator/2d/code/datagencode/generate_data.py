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
from training import get_f_phi, get_initial_condition, GaussianRF
from flux import Flux
from helper import f_to_DG, _evalf_2D_integrate, nabla
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from helper import legendre_inner_product, inner_prod_with_legendre

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
    del inner_loop_steps
    key0 = jax.random.PRNGKey(seed)

    assert args.equation == "euler"
    f_poisson_bracket_dict = create_f_poisson_bracket_dict(args, flux)
    f_poisson_solve_dict = create_f_poisson_solve_dict(args)
    if args.diffusion_coefficient > 0.0:
        f_diffusion_dict = create_f_diffusion_dict(args)


    @partial(
        jit,
        static_argnums=(
            0,
            1,
            2,
            3,
            4,
            5,
        ),
    )
    def simulate(up, order, f_phi, f_poisson_bracket, f_diffusion, f_forcing_sim, a0, t0):
        nx = args.nx_max // up
        ny = args.ny_max // up
        dx = args.Lx / (nx)
        dy = args.Ly / (ny)
        dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
        nt = int(T / dt) + 1
        dt = T / nt

        a_data = simulate_2D(
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
            equation=args.equation,
            a_data=None,
            output=True,
            f_phi=f_phi,
            f_poisson_bracket=f_poisson_bracket,
            f_diffusion=f_diffusion,
            f_forcing=f_forcing_sim,
            rk=FUNCTION_MAP[args.runge_kutta],
            inner_loop_steps=1,
        )
        a_data = np.concatenate((a0[None], a_data))
        t_data = t0 + np.arange(nt) * dt
        return {"a": a_data, "t": t_data}

    for order in args.orders:
        for up in args.upsampling:
            nx = args.nx_max // up
            ny = args.ny_max // up
            dx = args.Lx / nx
            dy = args.Ly / ny
            nt = int(T / (args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1))) + 1
            create_dataset(args, N, data_dir, unique_id, up, order, nx, ny, nt)


    for order in args.orders:
        for up in args.upsampling:
            key = key0

            f_poisson_bracket = f_poisson_bracket_dict[order]
            f_poisson_solve = f_poisson_solve_dict[order][up]
            f_phi = lambda zeta, t: f_poisson_solve(zeta)
            if args.diffusion_coefficient > 0.0:
                f_diffusion = f_diffusion_dict[order]
            else:
                f_diffusion = None

            if args.is_forcing:
                ff = lambda x, y, t: 0.1 * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
                y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 2 * order + 1)
                f_forcing_sim = lambda zeta: y_term
            else:
                f_forcing_sim = None

            for n in range(N):

                key, subkey = jax.random.split(key)
                t0 = 0.0
                if args.initial_condition == "fourier_space":
                    GRF = GaussianRF(2, 256, alpha=2.5, tau=7)
                    a0 = np.asarray(GRF.sample(1)[0][:,:,None])
                    a0 = convert_DG_representation(
                        a0[None], order, 0, nx, ny, args.Lx, args.Ly, args.equation
                    )[0]

                else:
                    f_init = get_initial_condition(subkey, args)
                    a0 = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8)
                    a0 = convert_DG_representation(
                        a0[None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
                    )[0]

                t1 = time()
                data = simulate(up, order, f_phi, f_poisson_bracket, f_diffusion, f_forcing_sim, a0, t0)
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



def compute_percent_error(a1, a2):
    return np.linalg.norm(((a1[:,:,0]-a2[:,:,0]))) / np.linalg.norm((a2[:,:,0]))


#@partial(jit, static_argnums=(0, 1, 2, 3),)
def compute_error(args, nxs, cfls, Tf, Np, key):

    key1, _ = jax.random.split(key)
    t0 = 0.0
    flux = Flux.UPWIND
    order = args.order_max

    if args.initial_condition == "fourier_space":
        GRF = GaussianRF(2, 256, alpha=2.5, tau=7)
        a0 = np.asarray(GRF.sample(1)[0][:,:,None])
        a0_exact = convert_DG_representation(
            a0[None], args.order_max, 0, args.nx_max, args.ny_max, args.Lx, args.Ly, args.equation
        )[0]
    else:
        f_init = get_initial_condition(subkey, args)
        a0_exact = f_to_DG(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, f_init, t0, n = 8)


    f_poisson_exact = get_poisson_solver(
        args.poisson_dir, args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max
    )
    f_phi_exact = lambda zeta, t: f_poisson_exact(zeta)

    if args.diffusion_coefficient > 0.0:
        f_diffusion_exact = get_diffusion_func(args.order_max, args.Lx, args.Ly, args.diffusion_coefficient)
    else:
        f_diffusion_exact = None
    if args.is_forcing:
        leg_ip_exact = np.asarray(legendre_inner_product(args.order_max))
        ffe = lambda x, y, t: 0.1 * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
        y_term_exact = inner_prod_with_legendre(args.nx_max, args.ny_max, args.Lx, args.Ly, args.order_max, ffe, 0.0, n = 8)
        f_forcing_exact = lambda zeta: y_term_exact
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
            flux,
            equation=args.equation,
            output=False,
            f_poisson_bracket=f_poisson_bracket_exact,
            f_phi=f_phi_exact,
            f_diffusion=f_diffusion_exact,
            f_forcing=f_forcing_exact,
            rk=FUNCTION_MAP[args.runge_kutta],
        )
        return a_f, t_f


    ########
    # Store exact data at Np intervals 
    ########

    T_chunk = Tf / Np

    assert Np >= 1

    dx_min = args.Lx / (args.nx_max)
    dy_min = args.Ly / (args.ny_max)
    cfl_exact = 4.0
    dt_exact = cfl_exact * ((dx_min * dy_min) / (dx_min + dy_min)) / (2 * args.order_max + 1)
    nt_sim_exact = int(T_chunk // dt_exact) + 1
    dt_exact = T_chunk / nt_sim_exact

    a_exact_all = a0_exact[...,None]
    a_exact = a0_exact
    t_exact = t0

    for j in range(Np):
        a_exact, t_exact = sim_exact(a_exact, t_exact, nt_sim_exact, dt_exact)
        a_exact_all = np.concatenate((a_exact_all, a_exact[...,None]),axis=-1)


    errors = np.zeros((len(nxs), Np+1))

    for n, nx in enumerate(nxs):
        print("nx is {}".format(nx))
        ny = nx
        dx = args.Lx / (nx)
        dy = args.Ly / (ny)
        dt = cfls[n] * ((dx * dy) / (dx + dy)) / (2 * order + 1)
        nt_sim = int(T_chunk // dt) + 1
        dt = T_chunk / nt_sim



        f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
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
            ff = lambda x, y, t: 0.1 * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
            y_term = inner_prod_with_legendre(nx, ny, args.Lx, args.Ly, order, ff, 0.0, n = 8)
            f_forcing_sim = lambda zeta: y_term
        else:
            f_forcing_sim = None

        @partial(
            jit,
            static_argnums=(
                2
            ),
        )
        def simulate(a_i, t_i, nt, dt):
            
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


        a0 = convert_DG_representation(
            a0_exact[None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
        )[0]
        a_f, t_f = a0, t0

        a_e = convert_DG_representation(
            a_exact_all[...,0][None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
        )[0]

        errors = errors.at[n, 0].set(compute_percent_error(a_f, a_e))

        for j in range(Np):
            a_f, t_f = simulate(a_f, t_f, nt_sim, dt)
            a_e = convert_DG_representation(
                a_exact_all[...,j+1][None], order, args.order_max, nx, ny, args.Lx, args.Ly, args.equation
            )[0]
            errors = errors.at[n, j+1].set(compute_percent_error(a_f, a_e))

    return errors
