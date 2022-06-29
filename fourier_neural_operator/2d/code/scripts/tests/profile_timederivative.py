import sys
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/simcode")
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/analysiscode")
sys.path.append("/Users/nmcgreiv/research/thesis/DG-data/2d/code/scripts")
basedir = "/Users/nmcgreiv/research/thesis/DG-data/2d"

from timederivative import time_derivative_euler
from poissonbracket import get_poisson_bracket
from poissonsolver import get_poisson_solver
from basisfunctions import legendre_inner_product
from flux import Flux
import jax.numpy as np
import jax
from jax import jit
from functools import partial
from helper import f_to_DG
from time import sleep

def gaussian(x, y, t):
    xc, yc = Lx / 2, Ly / 2
    return np.exp(
        -75 * ((x - xc) ** 2 / Lx ** 2 + (y - yc) ** 2 / Ly ** 2)
    )

nx = ny = 48
Lx = Ly = 1.0
dx = Lx / nx
dy = Ly / ny
order = 1
t0 = 0.0

flux = Flux.UPWIND

leg_ip = np.asarray(legendre_inner_product(order))
denominator = leg_ip * dx * dy

f_poisson_solve = get_poisson_solver(basedir, nx, ny, Lx, Ly, order)
f_poisson_bracket = get_poisson_bracket(basedir, order, flux)

zeta = f_to_DG(nx, ny, Lx, Ly, order, gaussian, t0)

jf_poisson_solve = jit(f_poisson_solve)
jf_poisson_bracket = jit(f_poisson_bracket)

jax.profiler.start_trace("/tmp/tensorboard")


phi = jf_poisson_solve(zeta)
dadt = jf_poisson_bracket(zeta, phi) / denominator[None, None, :]
dadt.block_until_ready()

sleep(1)

phi = jf_poisson_solve(zeta)
dadt = jf_poisson_bracket(zeta, phi) / denominator[None, None, :]
dadt.block_until_ready()


jax.profiler.stop_trace()