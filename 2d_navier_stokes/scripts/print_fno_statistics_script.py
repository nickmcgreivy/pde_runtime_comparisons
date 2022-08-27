import jax.numpy as np
import numpy as onp
from time import time
from jax import config, jit
config.update("jax_enable_x64", True)
from functools import partial

from arguments import get_args
from initial_conditions import get_initial_condition_FNO
from flux import Flux
from rungekutta import FUNCTION_MAP
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from fv_and_pseudospectral_baselines import vorticity, get_step_func, simulate_fv_baseline, get_velocity, get_forcing, downsample_ux, downsample_uy
from simulate import simulate_2D
################
# PARAMETERS OF SIMULATION
################

Lx = 1.0
Ly = 1.0
order_max = 2
nx_max = 32
ny_max = 32
forcing_coefficient = 0.1
runge_kutta = "ssp_rk3"
nxs = [8]
N_compute_runtime = 5
N_test = 1 # change to 5 or 10
t0 = 0.0
T_chunk = 1.0

Tf = 50.0
cfl_safety = 10.0
Re = 1e-3
#Tf = 30.0
#cfl_safety = 6.0
#Re = 1e-4

diffusion_coefficient = 1/Re
Ne = int(Tf)

################
# END PARAMETERS
################

def compute_percent_error(a1, a2):
    return np.linalg.norm(((a1[:,:,0]-a2[:,:,0]))) / np.linalg.norm((a2[:,:,0]))


def get_forcing_FNO(order, nx, ny, Lx, Ly, forcing_coefficient):
	ff = lambda x, y, t: forcing_coefficient * (np.sin( 2 * np.pi * (x + y) ) + np.cos( 2 * np.pi * (x + y) ))
	y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
	return lambda zeta: y_term



def print_stats(args, nxs):

	##### COMPUTE RUNTIME

	a0 = get_initial_condition_FNO()

	order = 2
	flux = Flux.UPWIND
	for nx in nxs:
		ny = nx
		dx = Lx / (nx)
		dy = Ly / (ny)
		dt = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
		nt = int(Tf // dt) + 1
		dt = Tf / nt

		a_i = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly)[0]

		f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
		f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
		f_phi = lambda zeta, t: f_poisson_solve(zeta)
		f_diffusion = get_diffusion_func(order, Lx, Ly, diffusion_coefficient)
		f_forcing_sim = get_forcing_FNO(order, nx, ny, Lx, Ly, forcing_coefficient)

		
		@partial(jit, static_argnums=(2),)
		def simulate(a_i, t_i, nt, dt):
			return simulate_2D(
				a_i,
				t_i,
				nx,
				ny,
				Lx,
				Ly,
				order,
				dt,
				nt,
				f_poisson_bracket,
				f_phi,
				a_data=None,
				output=False,
				f_diffusion=f_diffusion,
				f_forcing=f_forcing_sim,
				rk=FUNCTION_MAP[runge_kutta],
			)
		

		times = onp.zeros(N_compute_runtime)
		a_final, _ = simulate(a_i, t0, nt, dt)
		a_final.block_until_ready()
		for n in range(N_compute_runtime):
			t1 = time()
			a_final, _ = simulate(a_i, t0, nt, dt)
			a_final.block_until_ready()
			t2 = time()
			times[n] = t2 - t1

		print("order = {}, Tf = {}, nx = {}".format(order, Tf, nx))
		print("runtimes: {}".format(times))


	#### COMPUTE ERRORS

	errors = onp.zeros((len(nxs), Ne+1))

	for _ in range(N_test):
		a0 = get_initial_condition_FNO()

		f_poisson_bracket_exact = get_poisson_bracket(args.poisson_dir, order_max, flux)
		f_poisson_solve_exact = get_poisson_solver(args.poisson_dir, nx_max, ny_max, Lx, Ly, order_max)
		f_phi_exact = lambda zeta, t: f_poisson_solve_exact(zeta)
		f_diffusion_exact = get_diffusion_func(order_max, Lx, Ly, diffusion_coefficient)
		f_forcing_exact = get_forcing_FNO(order_max, nx_max, ny_max, Lx, Ly, forcing_coefficient)

		@partial(jit,static_argnums=(2),)
		def sim_exact(a_i, t_i, nt, dt):
			return simulate_2D(
				a_i,
				t_i,
				nx_max,
				ny_max,
				Lx,
				Ly,
				order_max,
				dt,
				nt,
				f_poisson_bracket_exact,
				f_phi_exact,
				a_data=None,
				output=False,
				f_diffusion=f_diffusion_exact,
				f_forcing=f_forcing_exact,
				rk=FUNCTION_MAP[runge_kutta],
			)

		dx_min = Lx / (nx_max)
		dy_min = Ly / (ny_max)
		cfl_exact = 4.0
		dt_exact = cfl_exact * ((dx_min * dy_min) / (dx_min + dy_min)) / (2 * order_max + 1)
		nt_sim_exact = int(T_chunk // dt_exact) + 1
		dt_exact = T_chunk / nt_sim_exact

		a0_exact = convert_DG_representation(a0[None], order_max, 0, nx_max, ny_max, Lx, Ly)[0]
		a_exact_all = a0_exact[...,None]
		a_exact = a0_exact
		t_exact = t0

		for j in range(Ne):
			a_exact, t_exact = sim_exact(a_exact, t_exact, nt_sim_exact, dt_exact)
			print(np.mean(a_exact))
			a_exact_all = np.concatenate((a_exact_all, a_exact[...,None]),axis=-1)

		for n, nx in enumerate(nxs):
			ny = nx
			dx = Lx / (nx)
			dy = Ly / (ny)
			dt = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
			nt = int(T_chunk // dt) + 1
			dt = T_chunk / nt

			f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
			f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
			f_phi = lambda zeta, t: f_poisson_solve(zeta)
			f_diffusion = get_diffusion_func(order, Lx, Ly, diffusion_coefficient)
			f_forcing_sim = get_forcing_FNO(order, nx, ny, Lx, Ly, forcing_coefficient)

			@partial(jit, static_argnums=(2),)
			def simulate(a_i, t_i, nt, dt):
				return simulate_2D(
					a_i,
					t_i,
					nx,
					ny,
					Lx,
					Ly,
					order,
					dt,
					nt,
					f_poisson_bracket,
					f_phi,
					a_data=None,
					output=False,
					f_diffusion=f_diffusion,
					f_forcing=f_forcing_sim,
					rk=FUNCTION_MAP[runge_kutta],
				)

			a_f = convert_DG_representation(a0[None], order, 0, nx, ny, Lx, Ly)[0]
			t_f = t0

			for j in range(Ne+1):
				a_e = convert_DG_representation(a_exact_all[...,j][None], order, order_max, nx, ny, Lx, Ly)[0]
				print(np.mean(a_f))
				errors[n, j] += compute_percent_error(a_f, a_e) / N_test
				a_f, t_f = simulate(a_f, t_f, nt, dt)

	print(errors)
	print(np.mean(errors))

def main():
	args = get_args()

	from jax.lib import xla_bridge
	device = xla_bridge.get_backend().platform
	print(device)

	print_stats(args, nxs)

if __name__ == '__main__':
	main()
	
