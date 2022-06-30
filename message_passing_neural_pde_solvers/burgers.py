import jax.numpy as np
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
vmap_polyval = vmap(np.polyval, (0, None), -1)
import jax
from jax.config import config
config.update("jax_enable_x64", True)


from sympy import legendre, diff, integrate, symbols
from functools import lru_cache, partial
import numpy as onp
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import time, timeit

PI = np.pi


def upper_B(m, k):
    x = symbols("x")
    expr = x ** k * (x + 0.5) ** m
    return integrate(expr, (x, -1, 0))


def lower_B(m, k):
    x = symbols("x")
    expr = x ** k * (x - 0.5) ** m
    return integrate(expr, (x, 0, 1))


def A(m, k):
    x = symbols("x")
    expr = legendre(k, x) * x ** m
    return integrate(expr, (x, -1, 1)) / (2 ** (m + 1))


@lru_cache(maxsize=10)
def get_B_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(2 * p):
            res[m, k] = upper_B(m, k)
    for m in range(p):
        for k in range(2 * p):
            res[m + p, k] = lower_B(m, k)
    return res


@lru_cache(maxsize=10)
def get_inverse_B(p):
    B = get_B_matrix(p)
    return onp.linalg.inv(B)


@lru_cache(maxsize=10)
def get_A_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(p):
            res[m, k] = A(m, k)
            res[m + p, k + p] = A(m, k)
    return res


def get_b_coefficients(a):
    p = a.shape[1]
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    b = np.einsum("km,jm->jk", B_inv, rhs)
    return b


def recovery_slope(a, p):
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)[1, :]
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    slope = np.einsum("m,jm->j", B_inv, rhs)
    return slope


def ssp_rk3(a_n, t_n, F, dt):
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2))
    return a_3, t_n + dt


def _quad_two_per_interval(f, a, b, n=5):
    mid = (a + b) / 2
    return _fixed_quad(f, a, mid, n) + _fixed_quad(f, mid, b, n)


def _fixed_quad(f, a, b, n=5):
    assert isinstance(n, int) and n <= 8 and n > 0
    w = {
        1: np.asarray([2.0]),
        2: np.asarray([1.0, 1.0]),
        3: np.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: np.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: np.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: np.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: np.asarray(
            [
                0.1294849661688696932706,
                0.2797053914892766679015,
                0.38183005050511894495,
                0.417959183673469387755,
                0.38183005050511894495,
                0.279705391489276667901,
                0.129484966168869693271,
            ]
        ),
        8: np.asarray(
            [
                0.1012285362903762591525,
                0.2223810344533744705444,
                0.313706645877887287338,
                0.3626837833783619829652,
                0.3626837833783619829652,
                0.313706645877887287338,
                0.222381034453374470544,
                0.1012285362903762591525,
            ]
        ),
    }[n]

    xi_i = {
        1: np.asarray([0.0]),
        2: np.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: np.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: np.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: np.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: np.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: np.asarray(
            [
                -0.9491079123427585245262,
                -0.7415311855993944398639,
                -0.4058451513773971669066,
                0.0,
                0.4058451513773971669066,
                0.7415311855993944398639,
                0.9491079123427585245262,
            ]
        ),
        8: np.asarray(
            [
                -0.9602898564975362316836,
                -0.7966664774136267395916,
                -0.5255324099163289858177,
                -0.1834346424956498049395,
                0.1834346424956498049395,
                0.5255324099163289858177,
                0.7966664774136267395916,
                0.9602898564975362316836,
            ]
        ),
    }[n]

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return np.sum(wprime[:, None] * f(x_i), axis=0)


def inner_prod_with_legendre(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    
    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    ) 
    j = np.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = np.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(np.asarray([[1.]]), xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)

def inner_prod_with_f(f, t, nx, dx):
    x = (np.arange(nx) + 1/2) * dx
    return f(x, t) * dx


def training_forcing_func(key, NM, OM, AM, mink, maxk, L):
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


def _diffusion_term_1D_burgers(a, t, dx, nu):
    slope_right = recovery_slope(a, 1) / dx
    slope_left = np.roll(slope_right, 1)
    return nu * (slope_right - slope_left)


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
        dif_term = _diffusion_term_1D_burgers(a, t, dx, nu)
    else:
        dif_term = 0.0

    if forcing_func is not None:
        nx = a.shape[0]
        forcing_term = inner_prod_with_f(forcing_func, t, nx, dx)
        forcing_term = forcing_term
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

###################################################################################################################
###################################################################################################################
# IGNORE ABOVE THIS LINE
###################################################################################################################
###################################################################################################################

#######################
# Runtime Hyperparameters
#######################


t0 = 0.0
Tf = 16.0
L = 16.0
nxs = [40, 50, 100]
nu = 0.0
cfl_safety_factors = [1.0, 1.0, 1.0]

flux="weno"

NM = 5
OM = 0.4
AM = 0.5
mink = 1 
maxk = 3

MAX = 2.5

num_plot=2
T = Tf/(num_plot-1)

###################################################################################################################
###################################################################################################################
# PART 1: RUNTIME DEMO
###################################################################################################################
###################################################################################################################


################## 
# Generate initial conditions
##################

key = jax.random.PRNGKey(1)

k1, k2 = jax.random.split(key, 2)

f_forcing = training_forcing_func(k1, NM, OM, AM, mink, maxk, L)
f_init = lambda x, t: f_forcing(x, 0.0)

#f_forcing = None#training_forcing_func(k1, NM, OM, AM, mink, maxk, L)
#f_init = lambda x, t: np.sin(2 * PI * x / L)



##################
# End Generate Initial Conditions
##################



#######################
# Begin Simulate
#######################

for i, nx in enumerate(nxs):
    dx = L/nx

    a0 = map_f_to_FV(f_init, t0, nx, dx) # (nx, 1) array
    print("nx is {}".format(a0.shape[0]))

    @partial(jax.jit, static_argnums=(2, 3))
    def sim(a0, t0, T, cfl_safety_factor):
        nt = int(T / (cfl_safety_factor * dx / MAX)) + 1
        dt = T / nt
        print(nt)
        return simulate_1D(a0, t0, dx, dt, nt, nu, forcing_func=f_forcing, flux=flux)

    sim(a0, t0, T, cfl_safety_factors[i]) # jit-compile first 
    (af, tf) = print_runtime(sim)(a0, t0, T, cfl_safety_factors[i])

    ##################
    # End Simulate, Begin Plot
    ##################

    to_plot = [a0, af]

    fig, axs = plt.subplots(1, num_plot, sharex=True, sharey=True, squeeze=True, figsize=(8,8/4))
    for j in range(num_plot):
        plot_subfig(to_plot[j], axs[j], L, color="#ff5555",linewidth=1.0)
    axs[0].set_xlim([0, L])
    axs[0].set_ylim([-MAX, MAX])





###################################################################################################################
###################################################################################################################
# PART 2: Compute losses as a function of time
###################################################################################################################
###################################################################################################################


#######################
# Loss Hyperparameters
#######################

key = jax.random.PRNGKey(1)
k1, _ = jax.random.split(key)
t0 = 0.0
T = 16.0
L = 16.0
nu = 0.0
cfl_safety_factor = 1.0
cfl_safety_factor_exact = 1.0
MAX = 2.5

NM = 5
OM = 0.4
AM = 0.5
mink = 1 
maxk = 3


num_loss = 8

nx_base = 50
num_upsampling = 4
upsamplings = 2**np.arange(num_upsampling)
exact_upsampling = 2**num_upsampling



############################
# Compute losses
############################


nx_exact = nx_base * exact_upsampling
dx_exact = L / nx_exact
nt_exact = int(T / (cfl_safety_factor_exact * dx_exact / (MAX)))
dt_exact = T / nt_exact

print("Values of nx are {}, nx_exact is {}".format(nx_base * upsamplings, nx_exact))

assert (nt_exact % exact_upsampling == 0)

def loss(a1, a2):
    return np.mean((a1 - a2)**2)

def compute_a(f_init, upsampling, forcing_func):
    nx = nx_base * upsampling
    dx = L / nx
    a0 = map_f_to_FV(f_init, t0, nx, dx)
    dt = dt_exact / (upsampling / exact_upsampling) * (cfl_safety_factor / cfl_safety_factor_exact)
    nt = nt_exact * (upsampling / exact_upsampling) / (cfl_safety_factor / cfl_safety_factor_exact)
    return simulate_1D(a0, t0, dx, dt, nt, nu, output=True, forcing_func=forcing_func, flux=flux)

def compute_a_exact(f_init, forcing_func):
    a0_exact = map_f_to_FV(f_init, t0, nx_exact, dx_exact)
    return simulate_1D(a0_exact, t0, dx_exact, dt_exact, nt_exact, nu, output=True, forcing_func=forcing_func, flux=flux)


def downsample(a_exact, upsampling):
    UP = int(exact_upsampling / upsampling)
    T_UP = int(UP * (cfl_safety_factor / cfl_safety_factor_exact))
    a_exact_time_downsampled = a_exact[::T_UP]
    nt, nx_exact = a_exact_time_downsampled.shape
    return np.mean(a_exact_time_downsampled.reshape(nt, -1, UP), axis=2)

losses = onp.zeros(num_upsampling)
for _ in range(num_loss):
    k1, k2 = jax.random.split(k1, 2)
    f_forcing = training_forcing_func(k1, NM, OM, AM, mink, maxk, L)
    f_init = lambda x, t: f_forcing(x, 0.0)
    a_exact = compute_a_exact(f_init, f_forcing)
    for j, upsampling in enumerate(upsamplings):
        a = compute_a(f_init, upsampling, f_forcing)
        a_exact_downsampled = downsample(a_exact, upsampling)
        losses[j] += loss(a, a_exact_downsampled) / num_loss


fig2, axs2 = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=True, figsize=(8,8/4))

losses = losses * 250

axs2.loglog(upsamplings, losses)
print(losses)



plt.show()
