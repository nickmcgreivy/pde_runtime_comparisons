import jax
import jax.numpy as np
import numpy as onp
from jax import jit, vmap, config
import h5py
from functools import partial
from time import time
import jax_cfd.base as cfd
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
config.update("jax_enable_x64", True)

from arguments import get_args
from flux import Flux
from rungekutta import FUNCTION_MAP
from helper import f_to_DG, _evalf_2D_integrate, legendre_inner_product, inner_prod_with_legendre, convert_DG_representation, vorticity_to_velocity
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from fv_and_pseudospectral_baselines import vorticity, get_velocity, get_step_func, simulate_fv_baseline, get_forcing, downsample_ux, downsample_uy
from simulate import simulate_2D
from initial_conditions import f_init_MLCFD

PI = np.pi

################
# PARAMETERS OF SIMULATION
################

Lx = 2 * PI
Ly = 2 * PI
order_max = 2
nx_max = 128
ny_max = 128
Re = 1000
viscosity = 1/Re
forcing_coefficient = 1.0 
damping_coefficient = 0.1
runge_kutta = "ssp_rk3"
orders = [0, 1, 2]
nxs = [[32, 64, 128], [32, 48, 64], [16, 32, 48]]
nxs_fv_baseline = [32, 64, 128]
baseline_dt_reduction = 8.0
T_runtime = 1.0
N_compute_runtime = 5
N_test = 1 # change to 5 or 10
t0 = 0.0

cfl_safety = 0.5
density = 1.
max_velocity = 7




################
# END PARAMETERS
################

def get_nt_dt_DG(T, nx, ny, order):
	dx = Lx / (nx)
	dy = Ly / (ny)
	dt_i = (cfl_safety / baseline_dt_reduction) * ((dx * dy) / (dx + dy))
	nt = int(T // dt_i) + 1
	dt = T / nt
	return nt, dt


def get_dt_baseline(grid):
	return cfd.equations.stable_time_step(max_velocity, cfl_safety, viscosity, grid)

def get_grid(nx, ny):
	return grids.Grid((nx, ny), domain=((0, Lx), (0, Ly)))


def get_velocity_cfd(u_x, u_y):
  assert u_x.shape == u_y.shape
  bcs = boundaries.periodic_boundary_conditions(2)
  
  grid = grids.Grid(u_x.shape, domain=((0, Lx), (0, Ly)))
  u_x = grids.GridVariable(grids.GridArray(u_x, grid.cell_faces[0], grid=grid), bcs)
  u_y = grids.GridVariable(grids.GridArray(u_y, grid.cell_faces[1], grid=grid), bcs)
  return (u_x, u_y)


def get_forcing_dg(order, nx, ny):
	leg_ip = np.asarray(legendre_inner_product(order))
	ff = lambda x, y, t: -4 * (2 * PI / Ly) * np.cos(4 * (2 * PI / Ly) * y)
	y_term = inner_prod_with_legendre(nx, ny, Lx, Ly, order, ff, 0.0, n = 8)
	dx = Lx / nx
	dy = Ly / ny
	return lambda zeta: (y_term - dx * dy * damping_coefficient * zeta * leg_ip) * forcing_coefficient


def create_datasets(args, device):
	f = h5py.File(
		"{}/data/{}_fv.hdf5".format(args.read_write_dir, device),
		"w",
	)
	for nx in nxs_fv_baseline:
		dset_a = f.create_dataset(str(nx), (1,), dtype="float64")
	f.close()
	for o, order in enumerate(orders):
		f = h5py.File(
			"{}/data/{}_order{}.hdf5".format(args.read_write_dir, device, order),
			"w",
		)
		for nx in nxs[o]:
			dset_a = f.create_dataset(str(nx), (1,), dtype="float64")
		f.close()



def compute_runtime(args, random_seed, device, orders, nxs, nxs_fv_baseline, baseline_dt_reduction, Tf):

	create_datasets(args, device)

	key = jax.random.PRNGKey(random_seed)

	grid_max = get_grid(nxs_fv_baseline[-1], nxs_fv_baseline[-1])
	v0 = cfd.initial_conditions.filtered_velocity_field(key, grid_max, max_velocity, 4)
	ux0, uy0 = v0
	a0 = cfd.finite_differences.curl_2d(v0).data[:,:,None]

	for nx in nxs_fv_baseline:

		#### Nick's baseline hacked together ####
		
		ny = nx
		dx = Lx / (nx)
		dy = Ly / (ny)
		dt_i = (cfl_safety / baseline_dt_reduction) * ((dx * dy) / (dx + dy))
		nt = int(Tf // dt_i) + 1
		dt = Tf / nt
		DS_FAC = nxs_fv_baseline[-1] // nx
		u_x_init_ds = downsample_ux(ux0.array.data, DS_FAC)
		u_y_init_ds = downsample_uy(uy0.array.data, DS_FAC)
		v_init = get_velocity(Lx, Ly, u_x_init_ds, u_y_init_ds)


		f_forcing_baseline = get_forcing(Lx, Ly, forcing_coefficient, damping_coefficient, nx, ny)
		step_func = get_step_func(viscosity, dt, forcing=f_forcing_baseline)

		times = onp.zeros(N_compute_runtime)
		ux, uy = simulate_fv_baseline(v_init, step_func, nt)
		ux.array.data.block_until_ready()
		for n in range(N_compute_runtime):
			t1 = time()
			ux, uy = simulate_fv_baseline(v_init, step_func, nt)
			ux.array.data.block_until_ready()
			t2 = time()
			times[n] = t2 - t1

		print("FV baseline, Tf = {}, nx = {}".format(Tf, nx))
		print("runtimes: {}".format(times))

		#### JAX-CFD FV

		grid = get_grid(nx, ny)
		nt = int(Tf // get_dt_baseline(grid)) + 1
		dt = Tf / nt

		step_fn = jax.jit(cfd.funcutils.repeated(
	    cfd.equations.semi_implicit_navier_stokes(
	        density=density, 
	        viscosity=viscosity, 
	        forcing=forcings.simple_turbulence_forcing(
				grid,
				constant_magnitude=1,
				constant_wavenumber=4,
				linear_coefficient=-0.1,
				forcing_type='kolmogorov'), 
	        dt=dt, 
	        grid=grid),
	    steps=nt))
		rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, 1))

		v_init = get_velocity_cfd(u_x_init_ds, u_y_init_ds)
		
		_, trajectory = rollout_fn(v_init)
		trajectory[0].array.data.block_until_ready()
		times = onp.zeros(N_compute_runtime)
		for n in range(N_compute_runtime):
			t1 = time()
			_, trajectory = rollout_fn(v_init)
			trajectory[0].array.data.block_until_ready()
			t2 = time()
			times[n] = t2 - t1

		print("ML-CFD FV baseline, Tf = {}, nx = {}".format(Tf, nx))
		print("runtimes: {}".format(times))
		f = h5py.File(
			"{}/data/{}_fv.hdf5".format(args.read_write_dir, device),
			"r+",
		)
		f[str(nx)][0] = np.median(times)
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
			nt, dt = get_nt_dt_DG(Tf, order, nx, ny)

			f_poisson_bracket = get_poisson_bracket(args.poisson_dir, order, flux)
			f_poisson_solve = get_poisson_solver(args.poisson_dir, nx, ny, Lx, Ly, order)
			f_phi = lambda zeta, t: f_poisson_solve(zeta)
			f_diffusion = get_diffusion_func(order, Lx, Ly, viscosity)
			f_forcing_sim = get_forcing_dg(order, nx, ny)

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


			a_init = convert_DG_representation(a0[None], order, order_max, nx, ny, Lx, Ly)[0]


			times = onp.zeros(N_compute_runtime)
			a_final, _ = simulate(a_init, t0, nt, dt)
			a_final.block_until_ready()
			for n in range(N_compute_runtime):
				t1 = time()
				a_final, _ = simulate(a_init, t0, nt, dt)
				a_final.block_until_ready()
				t2 = time()
				times[n] = t2 - t1

			print("order = {}, Tf = {}, nx = {}".format(order, Tf, nx))
			print("runtimes: {}".format(times))
			f = h5py.File(
				"{}/data/{}_order{}.hdf5".format(args.read_write_dir, device, order),
				"r+",
			)
			f[str(nx)][0] = np.median(times)
			f.close()



def main():
	args = get_args()


	from jax.lib import xla_bridge
	device = xla_bridge.get_backend().platform
	print(device)

	random_seed = 42

	compute_runtime(args, random_seed, device, orders, nxs, nxs_fv_baseline, baseline_dt_reduction, T_runtime)
	

if __name__ == '__main__':
	main()