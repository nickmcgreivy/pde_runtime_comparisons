import sys

sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/simcode")
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/analysiscode")
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/scripts")

import jax.numpy as np
import matplotlib.pyplot as plt

from plot_data import plot_FE_basis, plot_DG_basis
from poissonsolver import get_poisson_solver as poisson_solver
from helper import f_to_DG, f_to_FE, f_to_source, _2d_fixed_quad
from basisfunctions import legendre_poly

from jax.config import config
config.update("jax_enable_x64", True)
basedir = "/Users/nmcgreiv/research/thesis/DG-data/2d"


def test_problem_1(nx, ny, Lx, Ly, order):
    def f(x, y, t):
        x = x - Lx / 2
        y = y - Ly / 2
        return np.exp(-10 * (x ** 2 + y ** 2))

    xi = f_to_source(nx, ny, Lx, Ly, order, f, 0.0, quad_func=_2d_fixed_quad, n=8)
    poisson_solve = poisson_solver(basedir, nx, ny, Lx, Ly, order)
    phi = poisson_solve(xi)
    plot_DG_basis(nx, ny, Lx, Ly, order, xi, title="DG xi")
    plot_FE_basis(nx, ny, Lx, Ly, order, phi, title="Poisson phi")


def test_problem_2(nx, ny, Lx, Ly, order):
    def f(x, y, t):
        x = x - Lx / 2
        y = y - Ly / 2
        return np.exp(-10 * (2 * x ** 2 + 4 * x * y + 5 * y ** 2))

    xi = f_to_source(nx, ny, Lx, Ly, order, f, 0.0)
    poisson_solve = poisson_solver(basedir, nx, ny, Lx, Ly, order)
    phi = poisson_solve(xi)
    plot_DG_basis(nx, ny, Lx, Ly, order, xi, title="DG xi")
    plot_FE_basis(nx, ny, Lx, Ly, order, phi, title="Poisson phi")


def test_problem_3(nx, ny, Lx, Ly, order):
    x_1, y_1 = 0.35 * Lx, 0.5 * Ly
    x_2, y_2 = 0.65 * Lx, 0.5 * Ly
    f = lambda x, y, t: np.exp(-100 * ((x - x_1) ** 2 + (y - y_1) ** 2) / 0.8) + np.exp(
        -100 * ((x - x_2) ** 2 + (y - y_2) ** 2) / 0.8
    )
    xi = f_to_source(nx, ny, Lx, Ly, order, f, 0.0)
    poisson_solve = poisson_solver(basedir, nx, ny, Lx, Ly, order)
    phi = poisson_solve(xi)
    plot_DG_basis(nx, ny, Lx, Ly, order, xi, title="DG xi")
    plot_FE_basis(nx, ny, Lx, Ly, order, phi, title="Poisson phi")


def test_problem_4(nx, ny, Lx, Ly, order):
    Lx, Ly = 1, 1
    f = lambda x, y, t: (x - Lx / 2)
    f_exact = lambda x, y, t: -(x ** 3 / 6 - Lx * x ** 2 / 4 + 1 / 12 * Lx ** 2 * x)
    phi_exact = f_to_FE(nx, ny, Lx, Ly, order, f_exact, 0.0)
    xi = f_to_source(nx, ny, Lx, Ly, order, f, 0.0)
    poisson_solve = poisson_solver(basedir, nx, ny, Lx, Ly, order)
    phi = poisson_solve(xi)
    plot_DG_basis(nx, ny, Lx, Ly, order, xi, title="DG xi")
    plot_FE_basis(nx, ny, Lx, Ly, order, phi, title="Poisson phi")
    plot_FE_basis(nx, ny, Lx, Ly, order, phi_exact, title="Exact phi")


def test_timing(nx, ny, Lx, Ly, order):
    t1 = time.time()
    N_global_elements, M = load_assembly_matrix(basedir, nx, ny, order)
    t2 = time.time()
    print("Time to create assembly matrix: {}".format(t2 - t1))
    V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
    t3 = time.time()
    print("Time to create volume matrix: {}".format(t3 - t2))
    S = create_source_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
    t4 = time.time()
    print("Time to create source matrix: {}".format(t4 - t3))
    lu, piv = lu_factor_np(-V)
    t5 = time.time()
    print("Time to LU factor: {}".format(t5 - t4))
    print("Total time: {}".format(t5 - t1))


def test(N, o):
    Lx = 1
    Ly = 1
    test_timing(N, N, Lx, Ly, o)


def main():

    nx = 32
    ny = 32
    Lx = 1.0
    Ly = 2.0
    order = 1
    test_problem_1(nx, ny, Lx, Ly, order)
    test_problem_2(nx, ny, Lx, Ly, order)
    test_problem_3(nx, ny, Lx, Ly, order)
    # test_problem_4(nx, ny, Lx, Ly, order)
    plt.show()


if __name__ == "__main__":
    main()


# test(100, 1)
# test(200, 1)
# test(300, 1)
# test(400, 1)
# test(600, 1)
