import jax.numpy as np

def time_derivative_advection(
    zeta, t, f_poisson_bracket, f_phi, denominator, model=None, params=None, f_forcing=None, f_diffusion = None,
):
    phi = f_phi(zeta, t)
    if f_forcing is not None:
        forcing_term = f_forcing(zeta)
    else:
        forcing_term = 0.0
    if f_diffusion is not None:
        diffusion_term = f_diffusion(zeta)
    else:
        diffusion_term = 0.0
    return (
        (f_poisson_bracket(zeta, phi, model=model, params=params) + forcing_term + diffusion_term)
        / denominator[None, None, :]
    )


def time_derivative_euler(
    zeta, t, f_poisson_bracket, f_phi, denominator, model=None, params=None, f_forcing=None, f_diffusion = None,
):
    phi = f_phi(zeta, t)
    if f_forcing is not None:
        forcing_term = f_forcing(zeta)
    else:
        forcing_term = 0.0
    if f_diffusion is not None:
        diffusion_term = f_diffusion(zeta)
    else:
        diffusion_term = 0.0
    

    return (
        (f_poisson_bracket(zeta, phi, model=model, params=params) + forcing_term + diffusion_term)
        / denominator[None, None, :]
    )


def time_derivative_hw(
    a,
    t,
    f_poisson_bracket,
    f_phi,
    f_diff,
    f_deriv_y,
    denominator,
    alpha=0.0,
    kappa=0.0,
    f_forcing=None,
    f_diffusion = None,
    model=None,
    params=None,
):
    """
    Computes the time derivative da/dt = ... where

    a = {zeta, n} is a (2, nx, ny, num_elem) array,

    d zeta / dt + {phi, zeta} = alpha (phi - n) - D nabla^4 zeta

    d n / dt + {phi, n} = alpha (phi - n) - kappa d phi / dy - D nabla^4 n
    """
    zeta, n = a
    phi = f_phi(zeta, t)
    pb_zeta = f_poisson_bracket(zeta, phi, model=model, params=params)
    pb_n = f_poisson_bracket(n, phi, model=model, params=params)
    alpha_term = alpha * f_diff(phi, n)
    kappa_term = kappa * f_deriv_y(phi)
    if f_forcing is not None:
        raise NotImplementedError
    if f_diffusion is not None:
        diffusion_term_zeta = f_diffusion(zeta)
        diffusion_term_n = f_diffusion(n)
    else:
        diffusion_term_zeta = 0.0
        diffusion_term_n = 0.0
    dzeta_dt = (pb_zeta + alpha_term + diffusion_term_zeta) / denominator[None, None, :]
    dn_dt = (pb_n + alpha_term - kappa_term + diffusion_term_n) / denominator[None, None, :]
    return np.concatenate((dzeta_dt[None], dn_dt[None]))
