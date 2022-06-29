import jax.numpy as np
from jax.lax import scan
from functools import partial
from jax import vmap, jit, config, checkpoint

from timederivative import (
    time_derivative_advection,
    time_derivative_euler,
)
from rungekutta import ssp_rk3
from basisfunctions import legendre_inner_product
from poissonbracket import (
    get_diff,
    get_deriv_y,
)
from helper import inner_prod_with_legendre
from flux import Flux

PI = np.pi

def _scan(sol, x, rk_F):
    """
    Helper function for jax.lax.scan, which will evaluate f by stepping nt timesteps
    """
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None

def _scan_output(sol, x, rk_F):
    """
    Helper function for jax.scan, same as _scan but outputs data
    """
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), a_f

def _scan_output_inner_loop(sol, x, rk_F, inner_loop_steps):
    a, t = sol
    def f_scan(sol, x):
        a, t = sol
        return rk_F(a, t), None
    (a_f, t_f), _ = scan(f_scan, (a, t), None, length=inner_loop_steps)
    return (a_f, t_f), a_f

def _scan_loss(sol, a_exact, rk_F, f_loss):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), f_loss(a_f, a_exact)


def _scan_loss_inner_loop(sol, a_exact, rk_F, loss, inner_loop_steps):
    a, t = sol
    def f_scan(sol, x):
        a, t = sol
        return rk_F(a, t), None
    (a_f, t_f), _ = scan(f_scan, (a, t), None, length=inner_loop_steps)
    return (a_f, t_f), f_loss(a_f, a_exact)

def simulate_2D(
    a0,
    t0,
    nx,
    ny,
    Lx,
    Ly,
    order,
    dt,
    nt,
    flux,
    model=None,
    params=None,
    equation="advection",
    a_data=None,
    output=False,
    f_phi=lambda zeta, t: 0.0,
    f_diffusion=None,
    f_forcing=None,
    f_poisson_bracket=lambda zeta, phi: 0.0,
    rk=ssp_rk3,
    square_root_loss=False,
    mean_loss=True,
    inner_loop_steps=1,
):
    dx = Lx / nx
    dy = Ly / ny
    leg_ip = np.asarray(legendre_inner_product(order))
    denominator = leg_ip * dx * dy

    if equation == "advection":
        dadt = lambda a, t: time_derivative_advection(
            a, t, f_poisson_bracket, f_phi, denominator, 
            model=model, 
            params=params,
            f_forcing=f_forcing,
            f_diffusion=f_diffusion,
        )
    elif equation == "euler":
        dadt = lambda a, t: time_derivative_euler(
            a, t, f_poisson_bracket, f_phi, denominator, 
            model=model,
            params=params, 
            f_forcing=f_forcing,
            f_diffusion=f_diffusion,
        )
    else:
        raise NotImplementedError

    f_rk = lambda a, t: rk(a, t, dadt, dt)

    def MSE(a, a_exact):
        return np.mean(np.sum((a - a_exact) ** 2 / (leg_ip[None, None, :]), axis=-1))

    def MSE_sqrt(a, a_exact):
        return np.sqrt(MSE(a, a_exact))

    if square_root_loss:
        loss = MSE_sqrt
    else:
        loss = MSE

    if a_data is not None:
        assert nt == a_data.shape[0]

        @checkpoint
        @jit
        def scanfloss(sol, a_exact):
            return _scan_loss(sol, a_exact, f_rk, loss)

        @checkpoint
        @jit
        def scanflossinnerloop(sol, a_exact):
            return _scan_loss_inner_loop(sol, a_exact, f_rk, loss, inner_loop_steps)

        if inner_loop_steps > 1:
            scanf = scanflossinnerloop
        elif inner_loop_steps == 1:
            scanf = scanfloss
        else:
            raise Exception
        (a_f, t_f), loss = scan(scanf, (a0, t0), a_data)
        if mean_loss:
            return np.mean(loss)
        else:
            return loss
    else:
        if output:
            if inner_loop_steps > 1:
                scanf = jit(lambda sol, x: _scan_output_inner_loop(sol, x, f_rk, inner_loop_steps))
            elif inner_loop_steps == 1:
                scanf = jit(lambda sol, x: _scan_output(sol, x, f_rk))
            else:
                raise Exception
            _, data = scan(scanf, (a0, t0), None, length=nt)
            return data
        else:
            scanf = jit(lambda sol, x: _scan(sol, x, f_rk))
            (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
            return (a_f, t_f)
