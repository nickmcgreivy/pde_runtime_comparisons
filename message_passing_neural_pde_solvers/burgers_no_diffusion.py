import jax.numpy as np
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
vmap_polyval = vmap(np.polyval, (0, None), -1)
import jax
from jax.config import config
from sympy import legendre, diff, integrate, symbols
from functools import lru_cache, partial
import numpy as onp
from scipy.special import eval_legendre
import time, timeit
config.update("jax_enable_x64", True)

PI = np.pi

def ssp_rk3(a_n, t_n, F, dt):
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2))
    return a_3, t_n + dt


def inner_prod_with_f(f, t, nx, dx):
    x = (np.arange(nx) + 1/2) * dx
    return f(x, t) * dx


def forcing_func(key, NM, OM, AM, mink, maxk, L):
    key1, key2, key3, key4 = jax.random.split(key, 4)
    phases = jax.random.uniform(key1, (NM,)) * 2 * PI
    ks = jax.random.randint(key2, (NM,), mink, maxk + 1)
    amplitudes = (jax.random.uniform(key3, (NM,)) - 0.5) * 2 * AM
    omegas = (jax.random.uniform(key4, (NM,)) - 0.5) * 2 * OM * 2 * PI

    def sum_modes(x, t):
        return np.sum(
            amplitudes[None, :]
            * np.sin(
                2 * PI * ks[None, :] / L * (x[:, None]) + omegas[None, :] * t
                + phases[None, :]
            ),
            axis=1,
        )

    return sum_modes


def map_f_to_FV(f, t, nx, dx):
    return inner_prod_with_f(f, t, nx, dx) / dx


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), a_f


def _godunov_flux_1D_burgers(a):
    a = np.pad(a, ((0, 1)), "wrap")
    u_left = a[:-1]
    u_right = a[1:]
    zero_out = 0.5 * np.abs(np.sign(u_left) + np.sign(u_right))
    compare = np.less(u_left, u_right)
    F = lambda u: u ** 2 / 2
    return compare * zero_out * np.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * np.maximum(F(u_left), F(u_right))

def f(a):
    return a**2/2

def _weno_flux_1D_burgers(a):
    epsilon = 1e-6
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10
    a_minus2 = np.roll(a, 2)
    a_minus1 = np.roll(a, 1)
    a_plus1 = np.roll(a, -1)
    a_plus2 = np.roll(a, -2)
    a_plus3 = np.roll(a, -3)

    f_a_minus2 = f(a_minus2)
    f_a_minus1 = f(a_minus1)
    f_a = f(a)
    f_a_plus1 = f(a_plus1)
    f_a_plus2 = f(a_plus2)
    f_a_plus3 = f(a_plus3)

    # Moving to right, a > 0, f_plus
    f0 = (2/6) * f_a_minus2 - (7/6) * f_a_minus1 + (11/6) * f_a
    f1 = (-1/6) * f_a_minus1 + (5/6) * f_a + (2/6) * f_a_plus1
    f2 = (2/6) * f_a + (5/6) * f_a_plus1 + (-1/6) * f_a_plus2
    beta0 = (13/12) * (f_a_minus2 - 2 * f_a_minus1 + f_a)**2 + (1/4) * (f_a_minus2 - 4 * f_a_minus1 + 3 * f_a)**2
    beta1 = (13/12) * (f_a_minus1 - 2 * f_a  + f_a_plus1)**2 + (1/4) * (- f_a_minus1 + f_a_plus1)**2
    beta2 = (13/12) * (f_a - 2 * f_a_plus1  + f_a_plus2)**2 + (1/4) * (3 * f_a - 4 * f_a_plus1  + f_a_plus2)**2
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    f_plus = (alpha0 * f0 + alpha1 * f1 + alpha2 * f2) / (alpha0 + alpha1 + alpha2)

    

    # Moving to left, a < 0, f_minus
    f0 = (2/6)  * f_a_plus3 - (7/6) * f_a_plus2 + (11/6) * f_a_plus1
    f1 = (-1/6) * f_a_plus2 + (5/6) * f_a_plus1 + (2/6)  * f_a
    f2 = (2/6)  * f_a_plus1 + (5/6) * f_a       + (-1/6) * f_a_minus1
    beta0 = (13/12) * (f_a_plus3 - 2 * f_a_plus2 + f_a_plus1)**2  + (1/4) * (     f_a_plus3 - 4 * f_a_plus2  + 3 * f_a_plus1)**2
    beta1 = (13/12) * (f_a_plus2 - 2 * f_a_plus1 + f_a)**2        + (1/4) * ( -   f_a_plus2                  +     f_a)**2
    beta2 = (13/12) * (f_a_plus1 - 2 * f_a       + f_a_minus1)**2 + (1/4) * ( 3 * f_a_plus1 - 4 * f_a        +     f_a_minus1)**2
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    f_minus = (alpha0 * f0 + alpha1 * f1 + alpha2 * f2) / (alpha0 + alpha1 + alpha2)



    
    compare = np.less(a, a_plus1)
    zero_out = 0.5 * np.abs(np.sign(a) + np.sign(a_plus1))
    return compare * zero_out * np.minimum(f_minus, f_plus) + (
        1 - compare
    ) * np.maximum(f_minus, f_plus)


def time_derivative_1D_burgers(
    a, t, dx, flux, nu, forcing_func=None,
):
    if flux == "godunov":
        flux_right = _godunov_flux_1D_burgers(a)
    elif flux == "weno":
        flux_right = _weno_flux_1D_burgers(a)
    else:
        raise Exception

    if nu > 0.0:
        raise Exception
    else:
        dif_term = 0.0

    if forcing_func is not None:
        forcing_term = inner_prod_with_f(forcing_func, t, a.shape[0], dx)
    else:
        forcing_term = 0.0

    flux_left = np.roll(flux_right, 1)
    flux_term = (flux_left - flux_right)
    return (flux_term + dif_term + forcing_term) / dx
    

def simulate_1D(
    a0,
    t0,
    dx,
    dt,
    nt,
    nu = 0.0,
    output=False,
    forcing_func=None,
    rk=ssp_rk3,
    flux="godunov"
):

    dadt = lambda a, t: time_derivative_1D_burgers(
        a,
        t,
        dx,
        flux,
        nu,
        forcing_func=forcing_func,
    )

    rk_F = lambda a, t: rk(a, t, dadt, dt)

    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), None, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
        return (a_f, t_f)


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    ##### for the purposes of debugging
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(np.asarray([[1.]]), xi)
        return np.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    dx = L / nx
    xjs = np.arange(nx) * L / nx
    xs = xjs[None, :] + np.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None), 1)
    subfig.plot(
        xs,
        vmap_eval(xs, a, np.arange(nx), dx),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return

def print_runtime(func):
    #### decorator that prints the simulation time
    def wrapper(*args, **kwargs):
        ti = time.time()
        output = func(*args, **kwargs)
        tf = time.time()
        print("time to simulate is {} microseconds".format(int(10**6 * (tf - ti))))
        return output
    return wrapper


#######################
# Runtime Hyperparameters
#######################

t0 = 0.0
T = 16.0
L = 16.0
nu = 0.0
nxs = [40, 50, 100]
cfl_safety_factors = [2.0, 2.0, 2.0]
NM = 5
OM = 0.4
AM = 0.5
mink = 1 
maxk = 3
MAX = 2.5

################## 
# Generate initial conditions
##################

key = jax.random.PRNGKey(1)
k1, _ = jax.random.split(key, 2)
f_forcing = forcing_func(k1, NM, OM, AM, mink, maxk, L)
f_init = lambda x, t: f_forcing(x, 0.0)

#######################
# Begin Simulate
#######################

fig, axs = plt.subplots(len(nxs), 2, sharex=True, sharey=True, squeeze=True, figsize=(8,8/2))

for i, nx in enumerate(nxs):
    dx = L/nx

    a0 = map_f_to_FV(f_init, t0, nx, dx)
    print("nx is {}".format(a0.shape))

    @partial(jax.jit, static_argnums=(2, 3))
    def sim(a0, t0, T, cfl_safety_factor):
        nt = int(T / (cfl_safety_factor * dx / MAX)) + 1
        dt = T / nt
        return simulate_1D(a0, t0, dx, dt, nt, nu, forcing_func=f_forcing)

    sim(a0, t0, T, cfl_safety_factors[i]) # jit-compile first 
    (af, tf) = print_runtime(sim)(a0, t0, T, cfl_safety_factors[i])

    # plot
    to_plot = [a0, af]
    for j in range(2):
        plot_subfig(to_plot[j][:,None], axs[i,j], L, color="#ff5555",linewidth=1.0)
    axs[i,0].set_xlim([0, L])
    axs[i,0].set_ylim([-MAX, MAX])

plt.show()