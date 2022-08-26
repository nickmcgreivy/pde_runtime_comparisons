import jax
import jax.numpy as np
import numpy as onp
from jax import jit, vmap, config
import h5py
from functools import partial
from time import time
config.update("jax_enable_x64", True)

from arguments import get_args
from flux import Flux
from rungekutta import FUNCTION_MAP
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from fv_and_pseudospectral_baselines import vorticity, get_step_func, simulate_fv_baseline, get_velocity, get_forcing, downsample_ux, downsample_uy
from simulate import simulate_2D

PI = np.pi

################
# PARAMETERS OF SIMULATION
################

Lx = 2 * PI
Ly = 2 * PI
order_max = 2
nx_max = 128
ny_max = 128
cfl_safety = 0.5
Re = 1000
diffusion_coefficient = 1/Re
forcing_coefficient = 1.0 
damping_coefficient = 0.1
runge_kutta = "ssp_rk3"
max_k = 5
min_k = 1
num_init_modes = 6
amplitude_max = 4.0
N_test = 5

################
# END PARAMETERS
################

def get_initial_condition(key):

	def sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y):
		return np.sum(
			amplitudes[None, :]
			* np.sin(
				ks_x[None, :] * 2 * PI / Lx * x[:, None] + phases_x[None, :]
			) * np.sin(
				ks_y[None, :] * 2 * PI / Ly * y[:, None] + phases_y[None, :]
			),
			axis=1,
		)

	key1, key2, key3, key4, key5 = jax.random.split(key, 5)
	phases_x = jax.random.uniform(key1, (num_init_modes,)) * 2 * PI
	phases_y = jax.random.uniform(key2, (num_init_modes,)) * 2 * PI
	ks_x = jax.random.randint(
		key3, (num_init_modes,), min_k, max_k
	)
	ks_y = jax.random.randint(
		key4, (num_init_modes,), min_k, max_k
	)
	amplitudes = jax.random.uniform(key5, (num_init_modes,)) * amplitude_max
	return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)


def get_forcing_dg(order, nx, ny, Lx, Ly, damping_coefficient, forcing_coefficient):
	leg_ip = np.asarray(legendre_inner_product(order))
	ff = lambda x, y, t: -4 * (2 * PI / Ly) * np.cos(4 * (2 * PI / Ly) * y)
	y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
	dx = Lx / nx
	dy = Ly / ny
	return lambda zeta: (y_term - dx * dy * damping_coefficient * zeta * leg_ip) * forcing_coefficient


def compute_runtime(args, random_seed, device, orders, nxs, nxs_baseline, baseline_dt_reduction, Tf):

	key = jax.random.PRNGKey(random_seed)
	key1, key2 = jax.random.split(key)
	f_init = get_initial_condition(key1)

	t0 = 0.0
	a0 = f_to_DG(nx_max, ny_max, Lx, Ly, order_max, f_init, t0, n = 8)

	######
	# start with fv baseline implementation
	######
	a_init = convert_DG_representation(a0[None], 0, order_max, nxs_baseline[-1], nxs_baseline[-1], Lx, Ly)[0]
	f_ps_exact = jit(get_poisson_solver(args.poisson_dir, nxs_baseline[-1], nxs_baseline[-1], Lx, Ly, 0))
	u_x_init, u_y_init = vorticity_to_velocity(Lx, Ly, a_init, f_ps_exact)

	for nx in nxs_baseline:
		
		ny = nx
		dx = Lx / (nx)
		dy = Ly / (ny)
		dt = (cfl_safety / baseline_dt_reduction) * ((dx * dy) / (dx + dy))
		nt = int(Tf // dt) + 1
		dt = Tf / nt
		f_forcing_baseline = get_forcing(Lx, Ly, forcing_coefficient, damping_coefficient, nx, ny)
		step_func = get_step_func(diffusion_coefficient, dt, forcing=f_forcing_baseline)


		DS_FAC = nxs_baseline[-1] // nx
		u_x_init_ds = downsample_ux(u_x_init, DS_FAC)
		u_y_init_ds = downsample_uy(u_y_init, DS_FAC)
		v_init = get_velocity(Lx, Ly, u_x_init_ds, u_y_init_ds)

		times = onp.zeros(N_test)
		ux, uy = simulate_fv_baseline(v_init, step_func, nt)
		ux.array.data.block_until_ready()
		for n in range(N_test):
			t1 = time()
			ux, uy = simulate_fv_baseline(v_init, step_func, nt)
			ux.array.data.block_until_ready()
			t2 = time()
			times[n] = t2 - t1

		print("FV baseline, Tf = {}, nx = {}".format(Tf, nx))
		print("runtimes: {}".format(times))
		f = h5py.File(
			"{}/data/{}_FVbaseline_nx{}.hdf5".format(args.read_write_dir, device, nx),
			"w",
		)
		f["runtime"] = np.median(times)
		f.close()

	###### 
	# loop through orders and nxs of DG baseline
	######
	for o, order in enumerate(orders):
		if order == 0:
			flux = Flux.VANLEER
		else:
			flux = Flux.UPWIND

		for nx in nxs[o]:
			print("nx is {}, order = {}".format(nx, order))
			ny = nx
			dx = Lx / (nx)
			dy = Ly / (ny)
			dt = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
			nt = int(Tf // dt) + 1
			dt = Tf / nt

			f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
			f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
			f_phi = lambda zeta, t: f_poisson_solve(zeta)
			f_diffusion = get_diffusion_func(order, Lx, Ly, diffusion_coefficient)
			f_forcing_sim = get_forcing_dg(order, nx, ny, Lx, Ly, damping_coefficient, forcing_coefficient)

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


			a_init = convert_DG_representation(a0[None], order, order_max, nx, ny, Lx, Ly)[0]


			times = onp.zeros(N_test)
			a_final, _ = simulate(a_init, t0, nt, dt)
			a_final.block_until_ready()
			for n in range(N_test):
				t1 = time()
				a_final, _ = simulate(a_init, t0, nt, dt)
				a_final.block_until_ready()
				t2 = time()
				times[n] = t2 - t1

			print("order = {}, Tf = {}, nx = {}".format(order, Tf, nx))
			print("runtimes: {}".format(times))
			f = h5py.File(
				"{}/data/{}_order{}_nx{}.hdf5".format(args.read_write_dir, device, order, nx),
				"w",
			)
			f["runtime"] = np.median(times)
			f.close()



def main():
	args = get_args()

	orders = [0, 1, 2]
	nxs = [[32, 64, 128], [32, 48, 64], [16, 32, 48]]
	nxs_baseline = [32, 64, 128]
	baseline_dt_reduction = 6.0
	Tf = 1.0


	from jax.lib import xla_bridge
	device = xla_bridge.get_backend().platform
	print(device)

	random_seed = 42

	compute_runtime(args, random_seed, device, orders, nxs, nxs_baseline, baseline_dt_reduction, Tf)
	

if __name__ == '__main__':
	main()