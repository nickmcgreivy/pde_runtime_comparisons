import jax.numpy as np
import jax
from jax import jit, config
from functools import partial
import h5py
config.update("jax_enable_x64", True)

from arguments import get_args
from flux import Flux
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from simulate import simulate_2D
from rungekutta import FUNCTION_MAP
from fv_and_pseudospectral_baselines import vorticity, get_step_func, simulate_fv_baseline, get_velocity, get_forcing, downsample_ux, downsample_uy


PI = np.pi

################
# PARAMETERS OF SIMULATION
################

Lx = 2 * PI
Ly = 2 * PI
order_max = 2
nx_max = 64
ny_max = 64
t0 = 0.0
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
orders = [0, 1, 2]
nxs = [[32, 64, 128], [32, 48, 64], [16, 32, 48, 64]]
nxs_baseline = [32, 64, 128]
baseline_dt_reductions = [4.0, 6.0, 8.0]
Np = 100
exact_flux = Flux.UPWIND


Tf = 2.0
burn_in_time = 1.0
N_test = 1 # change to 5 or 10

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


def create_corr_file(n, args):
	f = h5py.File(
		"{}/data/corr_run{}.hdf5".format(args.read_write_dir, n),
		"w",
	)
	dset_new = f.create_dataset("fv", (Np,), dtype="float64")
	dset_new = f.create_dataset("ps", (Np,), dtype="float64")
	dset_new = f.create_dataset("order0", (Np,), dtype="float64")
	dset_new = f.create_dataset("order1", (Np,), dtype="float64")
	dset_new = f.create_dataset("order2", (Np,), dtype="float64")
	f.close()

def write_corr_file(n, args, name, j, value):
	f = h5py.File(
		"{}/data/corr_run{}.hdf5".format(args.read_write_dir, n),
		"r+",
	)
	f[name][j] = value
	f.close()


def compute_corrcoef(n, args, key, orders, nxs, nxs_baseline, baseline_dt_reductions, Tf, Np):

	create_corr_file(n, args)

	f_init = get_initial_condition(key)
	a0 = f_to_DG(nx_max, ny_max, Lx, Ly, order_max, f_init, t0, n = 8)

	dx_min = Lx / nx_max
	dy_min = Ly / ny_max
	dt_exact = cfl_safety * ((dx_min * dy_min) / (dx_min + dy_min)) / (2 * order_max + 1)
	nt_burn_in = int(burn_in_time // dt_exact) + 1
	dt_burn_in = burn_in_time / nt_burn_in
	
	f_poisson_exact = get_poisson_solver(args.poisson_dir, nx_max, ny_max, Lx, Ly, order_max)
	f_phi_exact = lambda zeta, t: f_poisson_exact(zeta)
	f_diffusion_exact = get_diffusion_func(order_max, Lx, Ly, diffusion_coefficient)
	f_forcing_exact = get_forcing_dg(order_max, nx_max, ny_max, Lx, Ly, damping_coefficient, forcing_coefficient)
	f_poisson_bracket_exact = get_poisson_bracket(args.poisson_dir, order_max, exact_flux)

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

	a_burn_in_exact, t_burn_in = sim_exact(a0, t0, nt_burn_in, dt_burn_in)
	
	########
	# Store exact data at Np intervals 
	########

	T_chunk = Tf / Np
	nt_sim_exact = int(T_chunk // dt_exact) + 1
	dt_exact = T_chunk / nt_sim_exact

	a_exact_all = a_burn_in_exact[...,None]
	a_exact = a_burn_in_exact
	t_exact = t_burn_in
	for j in range(Np):
		a_exact, t_exact = sim_exact(a_exact, t_exact, nt_sim_exact, dt_exact)
		a_exact_all = np.concatenate((a_exact_all, a_exact[...,None]),axis=-1)



	def store_correlation(a_f, order, nx, ny, name, j):
		a_e = convert_DG_representation(a_exact_all[...,j][None], order, order_max, nx, ny, Lx, Ly)[0]
		M = np.concatenate([a_f[:,:,0].reshape(-1)[:,None], a_e[:,:,0].reshape(-1)[:,None]],axis=1)
		corrcoeff_j = np.corrcoef(M.T)[0,1]
		print("Run: {}, {}, nx = {}, T = {:.1f}, corr = {}".format(n, name, nx, j * T_chunk, corrcoeff_j))
		write_corr_file(n, args, name, j, corrcoeff_j)

	
	######
	# start with baseline implementation
	######
	for nx in nxs_baseline:
		for BASELINE_DT_REDUCTION in baseline_dt_reductions:
			print("baseline, nx is {}, reduction is {}".format(nx, BASELINE_DT_REDUCTION))
			
			ny = nx
			dx = Lx / (nx)
			dy = Ly / (ny)
			dt = (cfl_safety / BASELINE_DT_REDUCTION) * ((dx * dy) / (dx + dy))
			nt_chunk = int(T_chunk // dt) + 1
			dt = T_chunk / nt_chunk
			f_forcing_baseline = get_forcing(Lx, Ly, nx, ny)
			step_func = get_step_func(diffusion_coefficient, dt, forcing=f_forcing_baseline)

			a_burn_in = convert_DG_representation(a_burn_in_exact[None], 0, order_max, nxs_baseline[-1], nxs_baseline[-1], Lx, Ly)[0]
			f_ps_exact = jit(get_poisson_solver(args.poisson_dir, nxs_baseline[-1], nxs_baseline[-1], Lx, Ly, 0))
			u_x_burn_in, u_y_burn_in = vorticity_to_velocity(args, a_burn_in, f_ps_exact)


			DS_FAC = nxs_baseline[-1] // nx
			u_x_burn_in = downsample_ux(u_x_burn_in, DS_FAC)
			u_y_burn_in = downsample_uy(u_y_burn_in, DS_FAC)
			v_burn_in = get_velocity(args, u_x_burn_in, u_y_burn_in)
			v_f = v_burn_in

			
			for j in range(Np+1):
				a_f = shift_down_left(vorticity(v_f))[..., None]
				store_correlation(a_f, 0, nx, ny, "fv", j)
				v_f = simulate_baseline(v_f, step_func, nt_chunk)

	

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
			dx = Lx / (nx)
			dy = Ly / (ny)
			dt = cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)
			nt_chunk = int(T_chunk // dt) + 1
			dt = T_chunk / nt_chunk

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
			def simulate(a_i, t_i, nt, dt, params=None):
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
					flux,
					f_poisson_bracket,
					f_phi,
					a_data=None,
					output=False,
					f_diffusion=f_diffusion,
					f_forcing=f_forcing_sim,
					rk=FUNCTION_MAP[args.runge_kutta],
				)


			a_burn_in = convert_DG_representation(a_burn_in_exact[None], order, order_max, nx, ny, Lx, Ly)[0]
			a_f, t_f = a_burn_in, t_burn_in

			for j in range(Np+1):
				store_correlation(a_f, order, nx, ny, "order{}".format(order), j)
				a_f, t_f = simulate(a_f, t_f, nt_chunk, dt)


	


def main():
	args = get_args()

	random_seed = 42
	key = jax.random.PRNGKey(random_seed)

	for n in range(N_test):
		key, _ = jax.random.split(key)
		compute_corrcoef(n, args, key, orders, nxs, nxs_baseline, baseline_dt_reductions, Tf, Np)

	#nxs = [[32, 48, 64, 96, 128, 192, 256], [16, 24, 32, 48, 64, 96, 128], [16, 24, 32, 48, 64, 96]]
	#nxs_baseline = [32, 64, 128, 256, 512]
	




if __name__ == '__main__':
	main()