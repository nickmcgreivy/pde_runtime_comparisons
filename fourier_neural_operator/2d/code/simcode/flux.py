from enum import Enum
import jax.numpy as np


class Flux(Enum):
    """
    Flux is a subclass of Enum, which determines the flux that is used to compute
    the time-derivative of the equation.

    LEARNED is the data-driven discretization of the equation, called the "learned
    flux interpolation"
    """

    UPWIND = "upwind"
    CENTERED = "centered"
    VANLEER = "vanleer"
    CONSERVATION = "conservation"

    def __str__(self):
        return self.value


class NetworkOutput(Enum):
    """
    The learned flux interpolation will output either an approximation of
    f, or an approximation of df/dx.
    """

    F = 0
    DFDX = 1


def minmod(r):
    return np.maximum(0, np.minimum(1, r))


def minmod_2(z1, z2):
    s = 0.5 * (np.sign(z1) + np.sign(z2))
    return s * np.minimum(np.absolute(z1), np.absolute(z2))


def minmod_3(z1, z2, z3):
    s = (
        0.5
        * (np.sign(z1) + np.sign(z2))
        * np.absolute(0.5 * ((np.sign(z1) + np.sign(z3))))
    )
    return s * np.minimum(np.absolute(z1), np.minimum(np.absolute(z2), np.absolute(z3)))
