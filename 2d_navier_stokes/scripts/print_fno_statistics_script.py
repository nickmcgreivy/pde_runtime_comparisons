import jax.numpy as np
import numpy as onp
from functools import partial
from time import time
from jax import config, jit
import jax_cfd.base as cfd
config.update("jax_enable_x64", True)

from arguments import get_args
from initial_conditions import get_initial_condition_FNO
from flux import Flux
from rungekutta import FUNCTION_MAP
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from simulate import simulate_2D

################
# PARAMETERS OF SIMULATION
################

Lx = 1.0
Ly = 1.0
order_exact = 2
nx_exact = 64
ny_exact = 64
forcing_coefficient = 0.1
runge_kutta = "ssp_rk3"
nxs_dg = [8, 16]
t0 = 0.0

N_compute_runtime = 5
N_test = 5 # change to 5 or 10

t_runtime = 50.0
cfl_safety = 10.0
cfl_safety_exact = 5.0
Re = 1e3
#t_runtime = 30.0
#cfl_safety = 6.0
#Re = 1e4
viscosity = 1/Re
t_chunk = 1.0
outer_steps = int(t_runtime)

################
# END PARAMETERS
################


################
# HELPER FUNCTIONS
################


def compute_percent_error(a1, a2):
    return np.linalg.norm(((a1[:,:,0]-a2[:,:,0]))) / np.linalg.norm((a2[:,:,0]))

def concatenate_vorticity(v0, trajectory):
    return np.concatenate((v0[None], trajectory), axis=0)

def get_forcing_FNO(order, nx, ny):
    ff = lambda x, y, t: forcing_coefficient * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
    y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
    return lambda zeta: y_term

def get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T):
    dx = Lx / (nx)
    dy = Ly / (ny)
    dt_i = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
    inner_steps = int(T // dt_i) + 1
    dt = T / inner_steps
    return inner_steps, dt

def get_dg_step_fn(args, nx, ny, order, T, cfl_safety=cfl_safety):
    if order == 0:
        flux = Flux.VANLEER
    else:
        flux = Flux.UPWIND
    
    f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
    f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
    f_phi = lambda zeta, t: f_poisson_solve(zeta)
    f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
    f_forcing_sim = get_forcing_FNO(order, nx, ny)
    
    inner_steps, dt = get_inner_steps_dt_DG(nx, ny, order, cfl_safety, T)

    @jit
    def simulate(a_i):
        a, _ = simulate_2D(a_i, t0, nx, ny, Lx, Ly, order, dt, inner_steps, 
                           f_poisson_bracket, f_phi, a_data=None, output=False, f_diffusion=f_diffusion,
                            f_forcing=f_forcing_sim, rk=FUNCTION_MAP[runge_kutta])
        return a
    return simulate


def get_trajectory_fn(step_fn, outer_steps):
    rollout_fn = jit(cfd.funcutils.trajectory(step_fn, outer_steps))
    
    def get_rollout(v0):
        _, trajectory = rollout_fn(v0)
        return trajectory
    
    return get_rollout


def print_runtime(args):

    a0 = get_initial_condition_FNO()

    order = 2

    for nx in nxs_dg:
        ny = nx

        a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]

        step_fn = get_dg_step_fn(args, nx, ny, order, t_runtime)
        rollout_fn = get_trajectory_fn(step_fn, 1)

        a_final = rollout_fn(a_i)
        a_final.block_until_ready()
        times = onp.zeros(N_compute_runtime)
        for n in range(N_compute_runtime):
            t1 = time()
            a_final = rollout_fn(a_i)
            a_final.block_until_ready()
            t2 = time()
            times[n] = t2 - t1

        print("order = {}, t_runtime = {}, nx = {}".format(order, t_runtime, nx))
        print("runtimes: {}".format(times))


def print_errors(args):

    order = 2

    errors = onp.zeros((len(nxs_dg), outer_steps+1))

    for _ in range(N_test):
        a0 = get_initial_condition_FNO()

        a_i = convert_DG_representation(a0[None], order_exact, 0, nx_exact, ny_exact, Lx, Ly, n=8)[0]
        exact_step_fn = get_dg_step_fn(args, nx_exact, ny_exact, order_exact, t_chunk, cfl_safety = cfl_safety_exact)
        exact_rollout_fn = get_trajectory_fn(exact_step_fn, outer_steps)
        exact_trajectory = exact_rollout_fn(a_i)
        exact_trajectory = concatenate_vorticity(a_i, exact_trajectory)


        for n, nx in enumerate(nxs_dg):
            ny = nx

            a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly, n=8)[0]
            step_fn = get_dg_step_fn(args, nx, ny, order, t_chunk)
            rollout_fn = get_trajectory_fn(step_fn, outer_steps)
            trajectory = rollout_fn(a_i)
            trajectory = concatenate_vorticity(a_i, trajectory)

            for j in range(outer_steps+1):
                a_ex = convert_DG_representation(exact_trajectory[j][None], order, order_exact, nx, ny, Lx, Ly, n=8)[0]
                errors[n, j] += compute_percent_error(trajectory[j], a_ex) / N_test

    print(np.mean(errors, axis=-1))

def main():
    args = get_args()

    from jax.lib import xla_bridge
    device = xla_bridge.get_backend().platform
    print(device)

    print_runtime(args)
    print_errors(args)

if __name__ == '__main__':
    main()
    
