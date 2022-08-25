import jax.numpy as np
from jax import jit, config
import h5py
from functools import partial
from time import time
config.update("jax_enable_x64", True)

from arguments import get_args
from flux import Flux
from helper import f_to_DG
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

	key1, key2, key3, key4, key5 = random.split(key, 5)
	phases_x = random.uniform(key1, (num_init_modes,)) * 2 * PI
	phases_y = random.uniform(key2, (num_init_modes,)) * 2 * PI
	ks_x = random.randint(
		key3, (num_init_modes,), min_k, max_k
	)
	ks_y = random.randint(
		key4, (num_init_modes,), min_k, max_k
	)
	amplitudes = random.uniform(key5, (num_init_modes,)) * amplitude_max
	return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)


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
	a, order_new, order_high, nx_new, ny_new, Lx, Ly, n = 8
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
	return vmap_convert_repr(a)


def vorticity_to_velocity(Lx, Ly, a, f_poisson):
	H = f_poisson(a)
	nx, ny, _ = H.shape
	dx = Lx / nx
	dy = Ly / ny

	u_y = -(H[:,:,2] - H[:,:,3]) / dx
	u_x = (H[:,:,2] - H[:,:,1]) / dy
	return u_x, u_y


def get_forcing_dg(order, nx, ny, Lx, Ly, damping_coefficient, forcing_coefficient):
	leg_ip = np.asarray(legendre_inner_product(order))
	ff = lambda x, y, t: -4 * (2 * PI / Ly) * np.cos(4 * (2 * PI / Ly) * y)
	y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
	dx = Lx / nx
	dy = Ly / ny
	return lambda zeta: (y_term - dx * dy * damping_coefficient * zeta * leg_ip) * forcing_coefficient



def compute_runtime(args, random_seed, orders, nxs, nxs_baseline, baseline_dt_reduction, Tf):

	key = jax.random.PRNGKey(random_seed)
	key1, key2 = jax.random.split(key)
	f_init = get_initial_condition(key1, args)

	a0 = f_to_DG(nx_max, ny_max, Lx, Ly, order_max, f_init, t0, n = 8)
	t0 = 0.0

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
		v_init = get_velocity(u_x_init_ds, u_y_init_ds)

		
		### time 1
		t1 = time()
		_, _ = simulate_fv_baseline(v_init, step_func, nt)
		t2 = time()
		_, _ = simulate_fv_baseline(v_init, step_func, nt)
		t3 = time()
		_, _ = simulate_fv_baseline(v_init, step_func, nt)
		t4 = time()
		 _, _ = simulate_fv_baseline(v_init, step_func, nt)
		t5 = time()
		T1 = t2 - t1
		T2 = t3 - t2
		T3 = t4 - t3
		T4 = t5 - t4
		print("baseline, Tf = {}, nx = {}\ncompile time = {:.5f}\nsecond runtime = {:.5f}\nthird runtime = {:.5f}\nfourth runtime = {:.5f}\n".format(Tf, nx, T1, T2, T3, T4))
		T_fv_baseline = np.mean([T2, T3, T4])
		# TODO: save to file

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
			def simulate(a_i, t_i, nt, dt, params=None):
				model = None
				
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
					a_data=None,
					output=False,
					f_phi=f_phi,
					f_poisson_bracket=f_poisson_bracket,
					f_diffusion=f_diffusion,
					f_forcing=f_forcing_sim,
					rk=FUNCTION_MAP[runge_kutta],
					inner_loop_steps=1,
				)


			a_init = convert_DG_representation(
				a0[None], order, order_max, nx, ny, Lx, Ly
			)[0]

			t1 = time()
			_, _ = simulate(a_init, t0, nt, dt)
			t2 = time()
			_, _ = simulate(a_init, t0, nt, dt)
			t3 = time()
			_, _ = simulate(a_init, t0, nt, dt)
			t4 = time()
			_, _ = simulate(a_init, t0, nt, dt)
			t5 = time()
			T1 = t2 - t1
			T2 = t3 - t2
			T3 = t4 - t3
			T4 = t5 - t4
			print("order = {}, Tf = {}, nx = {}\ncompile time = {:.5f}\nsecond runtime = {:.5f}\nthird runtime = {:.5f}\nfourth runtime = {:.5f}\n".format(order, Tf, nx, T1, T2, T3, T4))
			T_dg_baseline = np.mean([T2, T3, T4])
			# TODO: save to file


def main():
	args = get_args()

	orders = [0, 1, 2]
	nxs = [[32, 64, 128], [32, 48, 64], [16, 32, 48]]
	nxs_baseline = [32, 64, 128]
	baseline_dt_reduction = 6.0
	Tf = 1.0

	random_seed = 42

	compute_runtime(args, random_seed, orders, nxs, nxs_baseline, baseline_dt_reduction, Tf)
	

if __name__ == '__main__':
	main()