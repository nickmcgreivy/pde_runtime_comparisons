from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as np
from typing import Sequence

from flux import Flux
from basisfunctions import (
    num_elements,
    zeta_right_minus_matrix,
    zeta_right_plus_matrix,
    zeta_top_minus_matrix,
    zeta_top_plus_matrix,
)


class LearnedStencil2D(nn.Module):

    """
    For a single set of DG coefficients, applies a NN to produce
    the stencil s_{i+\frac{1}{2}, j, l, k} and s_{i, j+\frac{1}{2}, l, k}

    Inputs
    zeta: (nx, ny, num_elements) array of DG coefficients


    Outputs
    learned_stencil: A (nx, S, p) or (nx, p, S, p) array of the finite-difference coefficients
                                                                     to compute F, i.e. the learned stencil
    """

    features: Sequence[int]
    kernel_size: int = 3
    kernel_out: int = 4
    width_even: int = 2
    width_odd: int = 1
    base_stencil: Flux = Flux.CENTERED
    order: int = 1

    def setup(self):
        self.num_elements = num_elements(self.order)
        self.conv = CNNPeriodic2D_RightTop(
            self.features,
            kernel_size=self.kernel_size,
            kernel_out=self.kernel_out,
            N_out=self.width_even
            * self.width_odd
            * self.num_elements
            * (self.order + 1),
        )
        R = zeta_right_minus_matrix(self.order).T
        L = zeta_right_plus_matrix(self.order).T
        T = zeta_top_minus_matrix(self.order).T
        B = zeta_top_plus_matrix(self.order).T

        base_stencil_R = np.zeros((2, self.num_elements, self.order + 1))
        base_stencil_R = base_stencil_R.at[0:, :].add(R / 2)
        self.base_stencil_R = base_stencil_R.at[1:, :].add(L / 2)

        base_stencil_T = np.zeros((2, self.num_elements, self.order + 1))
        base_stencil_T = base_stencil_T.at[0, :, :].add(T / 2)
        self.base_stencil_T = base_stencil_T.at[1, :, :].add(B / 2)

    def __call__(self, inputs):
        s_R, s_T = self.conv(inputs)
        s_R = s_R.reshape(
            *s_R.shape[:-1],
            self.width_even,
            self.width_odd,
            self.num_elements,
            self.order + 1
        )  # (nx, ny, w_E, w_O, num_elem, order+1)
        s_R = s_R.at[..., 0, 0].add(
            -np.mean(s_R[..., 0, 0], axis=(-1, -2))[..., None, None]
        )
        s_T = s_T.reshape(
            *s_T.shape[:-1],
            self.width_odd,
            self.width_even,
            self.num_elements,
            self.order + 1
        )  # (nx, ny, w_E, w_O, num_elem, order+1)
        s_T = s_T.at[..., 0, 0].add(
            -np.mean(s_T[..., 0, 0], axis=(-1, -2))[..., None, None]
        )

        s_R = s_R.at[
            ...,
            self.width_even // 2 - 1 : self.width_even // 2 + 1,
            (self.width_odd - 1) // 2,
            :,
            :,
        ].add(self.base_stencil_R)
        s_T = s_T.at[
            ...,
            (self.width_odd - 1) // 2,
            self.width_even // 2 - 1 : self.width_even // 2 + 1,
            :,
            :,
        ].add(self.base_stencil_T)
        return s_R, s_T


class CNNPeriodic2D_RightTop(nn.Module):
    """
    1D convolutional neural network which takes an array in (nx, ny, num_elements)
    and returns an array of size (nx, ny, N_out).

    The convolutional network has num_layers = len(features), with
    len(features) + 1 total convolutions. The last convolution outputs an
    array of size (nx, ny, N_out).
    """

    features: Sequence[int]
    kernel_size: int = 3  # should be odd
    kernel_out: int = 4  # should be even
    N_out: int = (
        2 * 1 * (4) * (1 + 1)
    )  # width_L * width_R * num_elements * (order+1), possibly plus one
    assert kernel_size % 2 == 1
    assert kernel_out % 2 == 0

    def setup(self):
        dtype = np.float64
        kernel_init = nn.initializers.lecun_normal(dtype=dtype)
        bias_init = lambda key, shape: np.zeros(shape, dtype=dtype)
        zeros_init = lambda key, shape: np.zeros(shape, dtype=dtype)
        self.layers = [
            nn.Conv(
                features=feat,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="VALID",
                kernel_init=kernel_init,
                bias_init=bias_init,
            )
            for feat in self.features
        ]
        self.output_R = nn.Conv(
            features=self.N_out,
            kernel_size=(self.kernel_out, self.kernel_size),
            padding="VALID",
            kernel_init=zeros_init,
            bias_init=bias_init,
        )
        self.output_T = nn.Conv(
            features=self.N_out,
            kernel_size=(self.kernel_size, self.kernel_out),
            padding="VALID",
            kernel_init=zeros_init,
            bias_init=bias_init,
        )

    def __call__(self, inputs):
        """
        inputs: zeta (nx, ny, num_elements) OR some arbitrary vector of shape (nx, ny, N)
        outputs: s_R, s_T (nx, ny, N_out), unconstrained
        """
        x = inputs
        for lyr in self.layers:
            x = np.pad(
                x,
                (
                    ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
                    ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
                    (0, 0),
                ),
                "wrap",
            )
            x = lyr(x)
            x = nn.relu(x)
        x_R = np.pad(
            x,
            (
                (self.kernel_out // 2 - 1, self.kernel_out // 2),
                ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
                (0, 0),
            ),
            "wrap",
        )
        x_T = np.pad(
            x,
            (
                ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
                (self.kernel_out // 2 - 1, self.kernel_out // 2),
                (0, 0),
            ),
            "wrap",
        )
        s_R = self.output_R(x_R)
        s_T = self.output_T(x_T)
        return s_R, s_T
