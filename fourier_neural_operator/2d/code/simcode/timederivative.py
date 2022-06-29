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