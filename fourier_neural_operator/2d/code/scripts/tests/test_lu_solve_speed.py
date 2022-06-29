import sys
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/simcode")
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/analysiscode")
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/scripts")
basedir = "/Users/nmcgreiv/research/thesis/DG-data/2d"
from timeit import timeit

from poissonsolver import load_assembly_matrix, create_source_matrix, load_lu_factor, create_volume_matrix
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
from jax.scipy.linalg import lu_solve
from jax.scipy.sparse.linalg import cg
from scipy.linalg import lu_solve as np_lu_solve
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from plot_data import plot_FE_basis, plot_DG_basis
import matplotlib.pyplot as plt
import eigensolve

from helper import f_to_DG
from jax.config import config
config.update("jax_enable_x64", True)



Lx = 1.0
Ly = 1.0

def print_time(func, N, *args):
	print("Function runtime is {} in ms".format(timeit('func(*args)', number=N, globals=globals())*1000/N))



###### Test 1: Is einsum or multiply faster? ######
###### ANSWER: einsum is faster ######

def get_poisson_solver_A(nx, ny, order):
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	print("nx is {}, N_global_elements is {}".format(nx, N_global_elements))
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_solve((lu, piv), b)
		res = res - jnp.mean(res)
		return res.at[M].get()

	return jit(solve)


def get_poisson_solver_C(nx, ny, order):
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)

	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res, _ = cg(-V, b)
		res = res - jnp.mean(res)
		return res.at[M].get()

	return jit(solve)



nxs = []
order = 1
N = 4

for nx in nxs:

	ny = nx

	def f(x, y, t):
		x = x - Lx / 2
		y = y - Ly / 2
		return jnp.exp(-10 * (2 * x ** 2 + 4 * x * y + 5 * y ** 2))

	xi = f_to_source(nx, ny, Lx, Ly, order, f, 0.0)
	print("Grid size is {} by {}".format(nx, nx))
	#f_solve_A = get_poisson_solver_A(nx, ny, order)
	f_solve_C = get_poisson_solver_C(nx, ny, order)
	#res_A = f_solve_A(xi)	
	#res_C = f_solve_C(xi)
	#assert np.allclose(res_A, res_C, atol=1e-5)
	print("Testing A")
	print("Function runtime is {} in ms".format(timeit('f_solve_A(xi).block_until_ready()', number=N, globals=globals())*1000/N))
	#print("Testing C")
	#print("Function runtime is {} in ms".format(timeit('f_solve_C(xi).block_until_ready()', number=N, globals=globals())*1000/N))


##### Test 2: is numpy lu_solve or jax lu_solve faster? #####

def get_solve_jnp(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve():
		b = jnp.zeros(piv.shape)
		res = lu_solve((lu, piv), b)
		return res
	return jit(solve)

def get_solve_np(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve():
		b = np.zeros(piv.shape)
		res = np_lu_solve((lu, piv), b)
		return res
	return solve

def get_solve_scipy(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sparse_r = csr_matrix(V)
	lu_r = splu(V_sparse_r)

	def solve():
		b = np.zeros(V.shape[0])
		res = lu_r.solve(b)
		return res
	return solve

def get_solve_eigen(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	solver = eigensolve.SparseSolver(csr_matrix(V))

	def solve():
		b = np.zeros(N_global_elements)
		return solver.solve(b)

	return solve

nxs = [16, 32, 48, 64, 96]
order = 1
N = 10
for nx in nxs:
	f_A = get_solve_jnp(nx, order)
	#f_B = get_solve_np(nx, order)
	f_C = get_solve_scipy(nx, order)
	f_D = get_solve_eigen(nx, order)
	f_A()
	#f_B()
	f_C()
	f_D()
	print("Grid size is {} by {}".format(nx, nx))
	print("Testing JAX jitted lu_solve")
	print("Function runtime is {} in ms".format(timeit('f_A().block_until_ready()', number=N, globals=globals())*1000/N))
	#print("Testing numpy lu_solve")
	#print("Function runtime is {} in ms".format(timeit('f_B()', number=N, globals=globals())*1000/N))
	print("Testing Sparse Scipy")
	print("Function runtime is {} in ms".format(timeit('f_C()', number=N, globals=globals())*1000/N))
	print("Testing Sparse Eigen")
	print("Function runtime is {} in ms".format(timeit('f_D()', number=N, globals=globals())*1000/N))


##### Test 3: is numpy lu_solve or scipy sparse lu_solve faster? #####


nxs = []
order = 1
N = 5
for nx in nxs:
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = np.asarray(create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements))

	V_sparse_r = csr_matrix(V)
	V_sparse_c = csc_matrix(V)

	lu_r = splu(V_sparse_r)
	lu_c = splu(V_sparse_c)

	b = np.zeros(N_global_elements)

	lu_r.solve(b)
	lu_c.solve(b)

	print("Grid size is {} by {}".format(nx, nx))
	print("Testing Row Sparse lu_solve")
	print("Function runtime is {} in ms".format(timeit('lu_r.solve(b)', number=N, globals=globals())*1000/N))
	print("Testing Column Sparse lu_solve")
	print("Function runtime is {} in ms".format(timeit('lu_c.solve(b)', number=N, globals=globals())*1000/N))

	f_B = get_solve_np(nx, order)
	f_B()
	print("Testing numpy lu_solve")
	print("Function runtime is {} in ms".format(timeit('f_B()', number=N, globals=globals())*1000/N))

	f_A = get_solve_jnp(nx, order)
	f_A()
	print("Testing jax lu_solve")
	print("Function runtime is {} in ms".format(timeit('f_A().block_until_ready()', number=N, globals=globals())*1000/N))


##### Test 4: What happens when you don't subtract off mean?



def get_solve_normal(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sparse_r = csr_matrix(V.T)
	lu_r = splu(V_sparse_r)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_r.solve(np.asarray(b))
		print("mean is {}".format(jnp.mean(res)))
		res = res - jnp.mean(res)
		return res[M], b
	return (solve)


def get_solve_no_subtract_xi(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	print(V - V.T)
	V_sparse_r = csr_matrix(V.T)
	lu_r = splu(V_sparse_r)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		#xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_r.solve(np.asarray(b))
		print("mean is {}".format(jnp.mean(res)))
		res = res - jnp.mean(res)
		return res[M], b
	return (solve)

def get_solve_no_subtract_res(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sparse_r = csr_matrix(V.T)
	lu_r = splu(V_sparse_r)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_r.solve(np.asarray(b))
		print("mean is {}".format(jnp.mean(res)))
		#res = res - jnp.mean(res)
		return res[M], b
	return (solve)

def get_solve_no_subtract_both(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sparse_r = csr_matrix(V.T)
	lu_r = splu(V_sparse_r)

	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		#xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_r.solve(np.asarray(b))
		print("mean is {}".format(jnp.mean(res)))
		#res = res - jnp.mean(res)
		return res[M], b
	return (solve)



nxs = []
order = 1
Lx = Ly = 1.0

for nx in nxs:
	ny = nx
	def f(x, y, t):
		x = x - Lx / 2
		y = y - Ly / 2
		return jnp.exp(-10 * (2 * x ** 2 + 4 * x * y + 5 * y ** 2))
	def gaussian(x, y, t):
		xc, yc = Lx / 2, Ly / 2
		return jnp.exp(
		    -75 * ((x - xc) ** 2 / Lx ** 2 + (y - yc) ** 2 / Ly ** 2)
		)

	xi = f_to_DG(nx, ny, Lx, Ly, order, gaussian, 0.0)

	solve1 = get_solve_normal(nx, order)
	solve2 = get_solve_no_subtract_xi(nx, order)
	solve3 = get_solve_no_subtract_res(nx, order)
	solve4 = get_solve_no_subtract_both(nx, order)

	phi1, b1 = solve1(xi)
	phi2, b2 = solve2(xi)
	phi3, b3 = solve3(xi)
	phi4, b4 = solve4(xi)
	print(np.sum(phi1))
	print(np.sum(phi2))
	print(np.sum(phi3))
	print(np.sum(phi4))
	plot_FE_basis(nx, ny, Lx, Ly, order, phi1, title="Normal")
	plot_FE_basis(nx, ny, Lx, Ly, order, phi2, title="No subtract xi")
	plot_FE_basis(nx, ny, Lx, Ly, order, phi3, title="No subtract res")
	plot_FE_basis(nx, ny, Lx, Ly, order, phi4, title="No subtract both")
	plt.show()


##### Test 5: What happens to gradient when you don't subtract off mean?



def get_solve_normal(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		print("mean of b is {}".format(jnp.mean(b)))
		res = lu_solve((lu, piv), b)
		print("mean of res is {}".format(jnp.mean(res)))
		res = res - jnp.mean(res)
		return res.at[M].get()
	return (solve)


def get_solve_no_subtract_xi(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		#xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_solve((lu, piv), b)
		res = res - jnp.mean(res)
		return res.at[M].get()
	return jit(solve)

def get_solve_no_subtract_res(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_solve((lu, piv), b)
		#res = res - jnp.mean(res)
		return res.at[M].get()
	return jit(solve)

def get_solve_no_subtract_both(nx, order):
	ny = nx
	N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
	S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	lu, piv = load_lu_factor(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	def solve(xi):
		#xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
		b = -jnp.einsum('ijkl,jkl->i', S, xi)
		res = lu_solve((lu, piv), b)
		#res = res - jnp.mean(res)
		return res.at[M].get()
	return jit(solve)



nxs = []
order = 2
Lx = Ly = 1.0

for nx in nxs:
	ny = nx
	def f(x, y, t):
		x = x - Lx / 2
		y = y - Ly / 2
		return jnp.exp(-10 * (2 * x ** 2 + 4 * x * y + 5 * y ** 2))
	def gaussian(x, y, t):
		xc, yc = Lx / 2, Ly / 2
		return jnp.exp(
		    -75 * ((x - xc) ** 2 / Lx ** 2 + (y - yc) ** 2 / Ly ** 2)
		)

	xi = f_to_DG(nx, ny, Lx, Ly, order, gaussian, 0.0)

	solve1 = grad(lambda xi: np.mean(get_solve_normal(nx, order)(xi)**4))
	solve2 = grad(lambda xi: np.mean(get_solve_no_subtract_xi(nx, order)(xi)**4))
	solve3 = grad(lambda xi: np.mean(get_solve_no_subtract_res(nx, order)(xi)**4))
	solve4 = grad(lambda xi: np.mean(get_solve_no_subtract_both(nx, order)(xi)**4))

	phi1 = solve1(xi)
	phi2 = solve2(xi)
	phi3 = solve3(xi)
	phi4 = solve4(xi)
	print(np.sum(phi1))
	print(np.sum(phi2))
	print(np.sum(phi3))
	print(np.sum(phi4))
	plot_DG_basis(nx, ny, Lx, Ly, order, phi1, title="Normal")
	plot_DG_basis(nx, ny, Lx, Ly, order, phi2, title="No subtract xi")
	plot_DG_basis(nx, ny, Lx, Ly, order, phi3, title="No subtract res")
	plot_DG_basis(nx, ny, Lx, Ly, order, phi4, title="No subtract both")
	plt.show()