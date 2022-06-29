import matplotlib.pyplot as plt
import jax
import numpy as np
import h5py
from functools import lru_cache
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import seaborn as sns

from basisfunctions import node_locations, legendre_poly, FE_poly, num_elements
from helper import (
    f_to_DG,
    f_to_FE,
    f_to_source,
    _trapezoidal_integration,
    _2d_fixed_quad,
)

plt.rcParams["text.usetex"] = True


def plot_FE_basis(
    nx, ny, Lx, Ly, order, phi, plot_lines=False, title="", plotting_density=4
):
    """
    Inputs:

    phi, (nx, ny, num_elem) matrix
    """
    factor = order * plotting_density + 1
    num_elem = phi.shape[-1]
    basis = FE_poly(order)
    x = np.linspace(-1, 1, factor + 1)[:-1] + 1 / factor
    y = np.linspace(-1, 1, factor + 1)[:-1] + 1 / factor

    basis_x = np.zeros((factor, factor, num_elem))
    for i in range(factor):
        for j in range(factor):
            for k in range(num_elem):
                basis_x[i, j, k] = basis[k].subs("x", x[i]).subs("y", y[j])

    Nx_plot = nx * factor
    Ny_plot = ny * factor
    output = np.zeros((Nx_plot, Ny_plot))
    for i in range(nx):
        for j in range(ny):
            output[
                i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
            ] = np.sum(basis_x * phi[i, j, None, None, :], axis=-1)

    fig, axs = plt.subplots(figsize=(5 * np.sqrt(Lx / Ly) + 1, np.sqrt(Ly / Lx) * 5))
    x_plot = np.linspace(0, Lx, Nx_plot + 1)
    y_plot = np.linspace(0, Ly, Ny_plot + 1)
    pcm = axs.pcolormesh(
        x_plot,
        y_plot,
        output.T,
        shading="flat",
        cmap=sns.cm.icefire,  # vmin=0, vmax=1
    )
    axs.contour(
        (x_plot[:-1] + x_plot[1:]) / 2,
        (y_plot[:-1] + y_plot[1:]) / 2,
        output.T,
        colors="black",
    )
    axs.set_xlim([0, Lx])
    axs.set_ylim([0, Ly])
    axs.set_xticks([0, Lx])
    axs.set_yticks([0, Ly])
    axs.set_title(title)
    fig.colorbar(pcm, ax=axs, extend="max")

    if plot_lines:
        fig, axs = plt.subplots(figsize=(5 * np.sqrt(Lx / Ly), np.sqrt(Ly / Lx) * 5))
        for j in range(0, Nx_plot, 10):
            axs.plot(x_plot[:-1], output.T[j, :])
        fig, axs = plt.subplots(figsize=(5 * np.sqrt(Lx / Ly), np.sqrt(Ly / Lx) * 5))
        for j in range(0, Ny_plot, 10):
            axs.plot(y_plot[:-1], output.T[:, j])


def plot_DG_basis(
    nx, ny, Lx, Ly, order, zeta, plot_lines=False, title="", plotting_density=4
):
    factor = order * plotting_density + 1
    num_elem = zeta.shape[-1]
    basis = legendre_poly(order)
    x = np.linspace(-1, 1, factor + 1)[:-1] + 1 / factor
    y = np.linspace(-1, 1, factor + 1)[:-1] + 1 / factor

    basis_x = np.zeros((factor, factor, num_elem))
    for i in range(factor):
        for j in range(factor):
            for k in range(num_elem):
                basis_x[i, j, k] = basis[k].subs("x", x[i]).subs("y", y[j])
    Nx_plot = nx * factor
    Ny_plot = ny * factor
    output = np.zeros((Nx_plot, Ny_plot))
    for i in range(nx):
        for j in range(ny):
            output[
                i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
            ] = np.sum(basis_x * zeta[i, j, None, None, :], axis=-1)
    fig, axs = plt.subplots(figsize=(5 * np.sqrt(Lx / Ly) + 1, np.sqrt(Ly / Lx) * 5))
    x_plot = np.linspace(0, Lx, Nx_plot + 1)
    y_plot = np.linspace(0, Ly, Ny_plot + 1)
    pcm = axs.pcolormesh(
        x_plot,
        y_plot,
        output.T,
        shading="flat",
        cmap=sns.cm.icefire,  # vmin=0, vmax=1
    )
    axs.set_xlim([0, Lx])
    axs.set_ylim([0, Ly])
    axs.set_xticks([0, Lx])
    axs.set_yticks([0, Ly])
    axs.set_title(title)
    fig.colorbar(pcm, ax=axs, extend="max")

    if plot_lines:
        fig, axs = plt.subplots(figsize=(5 * np.sqrt(Lx / Ly), np.sqrt(Ly / Lx) * 5))
        for j in range(0, Nx_plot, 10):
            axs.plot(x_plot[:-1], output[j, :])
        fig, axs = plt.subplots(figsize=(5 * np.sqrt(Lx / Ly), np.sqrt(Ly / Lx) * 5))
        for j in range(0, Ny_plot, 10):
            axs.plot(y_plot[:-1], output[:, j])


def get_minmax(args, n, directory, unique_id):
    max_val = float("-inf")
    min_val = float("inf")
    for j, up in enumerate(args.upsampling):
        for i, order in enumerate(args.orders):
            f = h5py.File(
                "{}/{}_up{}_order{}.hdf5".format(directory, unique_id, up, order),
                "r",
            )
            max_ij = np.max(
                f["a_data"][n, :, :, :, 0]
                + np.sum(np.abs(f["a_data"][n, :, :, :, 1:]), axis=-1)
            )
            min_ij = np.min(
                f["a_data"][n, :, :, :, 0]
                - np.sum(np.abs(f["a_data"][n, :, :, :, 1:]), axis=-1)
            )
            if max_ij > max_val:
                max_val = max_ij
            if min_ij < min_val:
                min_val = min_ij
    return min_val, max_val


def get_minmax_hw(args, n, directory, unique_id):
    max_val_z = float("-inf")
    min_val_z = float("inf")
    max_val_n = float("-inf")
    min_val_n = float("inf")
    for j, up in enumerate(args.upsampling):
        for i, order in enumerate(args.orders):
            f = h5py.File(
                "{}/{}_up{}_order{}.hdf5".format(directory, unique_id, up, order),
                "r",
            )
            max_ij_z = np.max(
                f["a_data"][n, :, 0, :, :, 0]
                + np.sum(np.abs(f["a_data"][n, :, 0, :, :, 1:]), axis=-1)
            )
            min_ij_z = np.min(
                f["a_data"][n, :, 1, :, :, 0]
                - np.sum(np.abs(f["a_data"][n, :, 0, :, :, 1:]), axis=-1)
            )
            if max_ij_z > max_val_z:
                max_val_z = max_ij_z
            if min_ij_z < min_val_z:
                min_val_z = min_ij_z

            max_ij_n = np.max(
                f["a_data"][n, :, 1, :, :, 0]
                + np.sum(np.abs(f["a_data"][n, :, 1, :, :, 1:]), axis=-1)
            )
            min_ij_n = np.min(
                f["a_data"][n, :, 1, :, :, 0]
                - np.sum(np.abs(f["a_data"][n, :, 1, :, :, 1:]), axis=-1)
            )
            if max_ij_n > max_val_n:
                max_val_n = max_ij_n
            if min_ij_n < min_val_n:
                min_val_n = min_ij_n

    return min_val_z, max_val_z, min_val_n, max_val_n


@lru_cache(maxsize=12)
def get_plotting_basis(order, factor):
    num_elem = num_elements(order)
    basis = legendre_poly(order)
    x = np.linspace(-1, 1, factor + 1)[:-1] + 1 / factor
    y = np.linspace(-1, 1, factor + 1)[:-1] + 1 / factor
    basis_x = np.zeros((factor, factor, num_elem))
    for i in range(factor):
        for j in range(factor):
            for k in range(num_elem):
                basis_x[i, j, k] = basis[k].subs("x", x[i]).subs("y", y[j])
    return basis_x


def plot_axs(
    zeta, axs, Lx, Ly, order, vmin, vmax, label, plotting_density=4, cmap=sns.cm.icefire,
):
    """
    zeta should be DG representation at a single timestep
    """
    nx, ny, num_elem = zeta.shape

    factor = plotting_density * order + 1
    output = np.zeros((nx * factor, ny * factor))
    plotting_basis = get_plotting_basis(order, factor)

    for i in range(nx):
        for j in range(ny):
            output[
                i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
            ] = np.sum(plotting_basis * zeta[i, j, None, None, :], axis=-1)

    x_plot = np.linspace(0, Lx, (nx * factor) + 1)
    y_plot = np.linspace(0, Ly, (ny * factor) + 1)

    return axs.pcolormesh(
        x_plot,
        y_plot,
        output.T,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label=label,
    )

def compute_entropy(args, n, directory, unique_id, label):
    mass_fig, mass_axs = plt.subplots(
        len(args.orders),
        len(args.upsampling),
        figsize=(12, 7.5),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    entropy_fig, entropy_axs = plt.subplots(
        len(args.orders),
        len(args.upsampling),
        figsize=(12, 7.5),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    for i, order in enumerate(args.orders):
        for j, up in enumerate(args.upsampling):
            f = h5py.File(
                "{}/{}_up{}_order{}.hdf5".format(directory, unique_id, up, order),
                "r",
            )
            NT = f["a_data"].shape[1]
            masses = np.zeros(NT)
            entropys = np.zeros(NT)
            times = f["t_data"][n] 
            for t in range(NT):

                if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                    raise NotImplementedError
                else:
                    zeta = f["a_data"][n, t]
                    mass = np.mean(zeta[:,:,0])
                    entropy = -np.mean(zeta[:,:,0]**2)
                    masses[t] = mass
                    entropys[t] = entropy
            mass_axs[i,j].plot(times, masses, label=label)
            entropy_axs[i,j].plot(times, entropys, label=label)
            entropy_axs[i,j].legend()
            mass_axs[i,j].legend()




def plot_data(args, n, directory, unique_id, label):

    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        vmin_z, vmax_z, vmin_n, vmax_n = get_minmax_hw(args, n, directory, unique_id)
    else:
        vmin_z, vmax_z = get_minmax(args, n, directory, unique_id)

    fig_i_z, axs_i_z = plt.subplots(
        len(args.orders),
        len(args.upsampling),
        figsize=(12, 7.5),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig_f_z, axs_f_z = plt.subplots(
        len(args.orders),
        len(args.upsampling),
        figsize=(12, 7.5),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        fig_i_n, axs_i_n = plt.subplots(
            len(args.orders),
            len(args.upsampling),
            figsize=(12, 7.5),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        fig_f_n, axs_f_n = plt.subplots(
            len(args.orders),
            len(args.upsampling),
            figsize=(12, 7.5),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

    for i, order in enumerate(args.orders):
        for j, up in enumerate(args.upsampling):
            f = h5py.File(
                "{}/{}_up{}_order{}.hdf5".format(directory, unique_id, up, order),
                "r",
            )
            axs_i_z[i, j].set_aspect(args.Ly / args.Lx)
            axs_f_z[i, j].set_aspect(args.Ly / args.Lx)
            if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                axs_i_n[i, j].set_aspect(args.Ly / args.Lx)
                axs_f_n[i, j].set_aspect(args.Ly / args.Lx)

            if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                zeta_i = f["a_data"][n, 0, 0]
                zeta_f = f["a_data"][n, -1, 0]
                n_i = f["a_data"][n, 0, 1]
                n_f = f["a_data"][n, -1, 1]
            else:
                zeta_i = f["a_data"][n, 0]
                zeta_f = f["a_data"][n, -1]

            im_i_z = plot_axs(
                zeta_i,
                axs_i_z[i, j],
                args.Lx,
                args.Ly,
                order,
                -2,
                2,
                label,
            )

            im_f_z = plot_axs(
                zeta_f,
                axs_f_z[i, j],
                args.Lx,
                args.Ly,
                order,
                -2,
                2,
                label,
            )

            if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                im_i_n = plot_axs(
                    n_i,
                    axs_i_n[i, j],
                    args.Lx,
                    args.Ly,
                    order,
                    vmin_n,
                    vmax_n,
                    label,
                )

                im_f_n = plot_axs(
                    n_f,
                    axs_f_n[i, j],
                    args.Lx,
                    args.Ly,
                    order,
                    vmin_n,
                    vmax_n,
                    label,
                )

    for j, up in enumerate(args.upsampling):
        axs_i_z[-1, j].set_xlabel("{}".format(up), fontsize=12)
        axs_f_z[-1, j].set_xlabel("{}".format(up), fontsize=12)
        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            axs_i_n[-1, j].set_xlabel("{}".format(up), fontsize=12)
            axs_f_n[-1, j].set_xlabel("{}".format(up), fontsize=12)
    for i, order in enumerate(args.orders):
        axs_i_z[i, 0].set_ylabel("{}".format(order), fontsize=12)
        axs_f_z[i, 0].set_ylabel("{}".format(order), fontsize=12)
        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            axs_i_n[i, 0].set_ylabel("{}".format(order), fontsize=12)
            axs_f_n[i, 0].set_ylabel("{}".format(order), fontsize=12)
    for i, order in enumerate(args.orders):
        for j, up in enumerate(args.upsampling):
            axs_i_z[i, j].set_xlim([0, args.Lx])
            axs_i_z[i, j].set_ylim([0, args.Ly])
            axs_i_z[i, j].set_xticks([0, args.Lx])
            axs_i_z[i, j].set_yticks([0, args.Ly])
            axs_f_z[i, j].set_xlim([0, args.Lx])
            axs_f_z[i, j].set_ylim([0, args.Ly])
            axs_f_z[i, j].set_xticks([0, args.Lx])
            axs_f_z[i, j].set_yticks([0, args.Ly])
            if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                axs_i_n[i, j].set_xlim([0, args.Lx])
                axs_i_n[i, j].set_ylim([0, args.Ly])
                axs_i_n[i, j].set_xticks([0, args.Lx])
                axs_i_n[i, j].set_yticks([0, args.Ly])
                axs_f_n[i, j].set_xlim([0, args.Lx])
                axs_f_n[i, j].set_ylim([0, args.Ly])
                axs_f_n[i, j].set_xticks([0, args.Lx])
                axs_f_n[i, j].set_yticks([0, args.Ly])
        fig_i_z.supxlabel("Downsampling factor")
        fig_i_z.supylabel("Highest polynomial degree")
        fig_f_z.supxlabel("Downsampling factor")
        fig_f_z.supylabel("Highest polynomial degree")
        fig_i_z.suptitle("Initial condition for vorticity $\zeta$")
        fig_f_z.suptitle("Final state for vorticity $\zeta$")
        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            fig_i_n.supxlabel("Downsampling factor")
            fig_i_n.supylabel("Highest polynomial degree")
            fig_f_n.supxlabel("Downsampling factor")
            fig_f_n.supylabel("Highest polynomial degree")
            fig_i_n.suptitle("Initial condition for density $n$")
            fig_f_n.suptitle("Final state for density $n$")
    fig_i_z.colorbar(im_i_z, ax=axs_i_z.ravel().tolist())
    fig_f_z.colorbar(im_f_z, ax=axs_f_z.ravel().tolist())
    # fig_i_z.tight_layout()
    # fig_f_z.tight_layout()
    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        fig_i_n.colorbar(im_i_n, ax=axs_i_n.ravel().tolist())
        fig_f_n.colorbar(im_f_n, ax=axs_f_n.ravel().tolist())
        # fig_i_n.tight_layout()
        # fig_f_n.tight_layout()


def movie_data(args, n, T, directory, unique_id, label, plotting_density=2, cmap=sns.cm.icefire):

    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        vmin_z, vmax_z, vmin_n, vmax_n = get_minmax_hw(args, n, directory, unique_id)
    else:
        vmin_z, vmax_z = get_minmax(args, n, directory, unique_id)

    def iter_frames():
        f = h5py.File(
            "{}/{}_up{}_order{}.hdf5".format(
                directory, unique_id, max(args.upsampling), min(args.orders)
            ),
            "r",
        )
        num_timesteps = f["a_data"].shape[1]
        NF = args.frames_per_time * T
        return np.floor(np.arange(NF) * num_timesteps / NF).astype(int)

    NU = len(args.upsampling)
    NP = len(args.orders)

    fig_z, axs_z = plt.subplots(NP, NU, figsize=(12, 7.5), squeeze=False)
    meshes_z = []
    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        fig_n, axs_n = plt.subplots(NP, NU, figsize=(12, 7.5), squeeze=False)
        meshes_n = []

    for j, up in enumerate(args.upsampling):
        for i, order in enumerate(args.orders):
            nx = args.nx_max // up
            ny = args.ny_max // up
            factor = plotting_density * order + 1
            x_plot = np.linspace(0, args.Lx, (nx * factor) + 1)
            y_plot = np.linspace(0, args.Ly, (ny * factor) + 1)
            output = np.zeros((nx * factor, ny * factor))
            meshes_z.append(
                axs_z[i, j].pcolormesh(
                    x_plot,
                    y_plot,
                    output.T,
                    shading="flat",
                    cmap=cmap,
                    vmin=vmin_z,
                    vmax=vmax_z,
                    label=label,
                )
            )
            axs_z[i, j].set_xlim([0.0, args.Lx])
            axs_z[i, j].set_ylim([0.0, args.Ly])
            axs_z[i, j].set_xticks([0.0, args.Lx])
            axs_z[i, j].set_yticks([0.0, args.Ly])
            axs_z[i, j].set_aspect(args.Ly / args.Lx)
            axs_z[i, j].set_aspect(args.Ly / args.Lx)
            if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                meshes_n.append(
                    axs_n[i, j].pcolormesh(
                        x_plot,
                        y_plot,
                        output.T,
                        shading="flat",
                        cmap=cmap,
                        vmin=vmin_n,
                        vmax=vmax_n,
                        label=label,
                    )
                )
                axs_n[i, j].set_xlim([0.0, args.Lx])
                axs_n[i, j].set_ylim([0.0, args.Ly])
                axs_n[i, j].set_xticks([0.0, args.Lx])
                axs_n[i, j].set_yticks([0.0, args.Ly])
                axs_n[i, j].set_aspect(args.Ly / args.Lx)
                axs_n[i, j].set_aspect(args.Ly / args.Lx)

    for j, up in enumerate(args.upsampling):
        axs_z[-1, j].set_xlabel("{}".format(up), fontsize=12)
        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            axs_n[-1, j].set_xlabel("{}".format(up), fontsize=12)
    for i, order in enumerate(args.orders):
        axs_z[i, 0].set_ylabel("{}".format(order), fontsize=12)
        if args.equation == "hw" or args.equation == "hasegawa_wakatani":
            axs_n[i, 0].set_ylabel("{}".format(order), fontsize=12)
    fig_z.supxlabel("Downsampling factor")
    fig_z.supylabel("Highest polynomial degree")
    fig_z.suptitle("Vorticity $\zeta$")
    fig_z.colorbar(meshes_z[0], ax=axs_z.ravel().tolist())
    # fig_z.tight_layout()
    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        fig_n.supxlabel("Downsampling factor")
        fig_n.supylabel("Highest polynomial degree")
        fig_n.suptitle("Density $n$")
        fig_n.colorbar(meshes_n[0], ax=axs_n.ravel().tolist())
        # fig_n.tight_layout()

    def init_z():
        for mesh in meshes_z:
            mesh.set_array([])
        return meshes_z

    def init_n():
        for mesh in meshes_n:
            mesh.set_array([])
        return meshes_z

    def set_data(zeta, mesh, order, plotting_density=2):
        nx, ny, num_elem = zeta.shape
        factor = plotting_density * order + 1
        output = np.zeros((nx * factor, ny * factor))
        plotting_basis = get_plotting_basis(order, factor)

        for i in range(nx):
            for j in range(ny):
                output[
                    i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
                ] = np.sum(plotting_basis * zeta[i, j, None, None, :], axis=-1)
        mesh.set_array(output.T.ravel())

    def animate_z(t):
        for i, order in enumerate(args.orders):
            for j, up in enumerate(args.upsampling):
                f = h5py.File(
                    "{}/{}_up{}_order{}.hdf5".format(directory, unique_id, up, order),
                    "r",
                )

                nt = int(
                    t
                    * max(args.upsampling)
                    / up
                    * (2 * order + 1)
                    / (2 * min(args.orders) + 1)
                )
                if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                    zeta = f["a_data"][n, nt, 0]
                else:
                    zeta = f["a_data"][n, nt]
                set_data(
                    zeta, meshes_z[j * NP + i], order, plotting_density=plotting_density
                )
        return meshes_z

    def animate_n(t):
        for i, order in enumerate(args.orders):
            for j, up in enumerate(args.upsampling):
                f = h5py.File(
                    "{}/{}_up{}_order{}.hdf5".format(directory, unique_id, up, order),
                    "r",
                )

                nt = int(
                    t
                    * max(args.upsampling)
                    / up
                    * (2 * order + 1)
                    / (2 * min(args.orders) + 1)
                )
                if args.equation == "hw" or args.equation == "hasegawa_wakatani":
                    density = f["a_data"][n, nt, 1]
                else:
                    raise Exception
                set_data(
                    density,
                    meshes_n[j * NP + i],
                    order,
                    plotting_density=plotting_density,
                )
        return meshes_n

    if args.equation == "hw" or args.equation == "hasegawa_wakatani":
        anim_z = animation.FuncAnimation(
            fig_z,
            animate_z,
            init_func=init_z,
            frames=iter_frames(),
            interval=args.movie_delay,
            blit=True,
        )
        anim_n = animation.FuncAnimation(
            fig_n,
            animate_n,
            init_func=init_n,
            frames=iter_frames(),
            interval=args.movie_delay,
            blit=True,
        )
        return anim_z, anim_n
    else:
        anim_z = animation.FuncAnimation(
            fig_z,
            animate_z,
            init_func=init_z,
            frames=iter_frames(),
            interval=args.movie_delay,
            blit=True,
        )
        return anim_z


def analyze_data(args, Np, T, directory, unique_id, label):
    key = jax.random.PRNGKey(args.random_seed)
    f = h5py.File(
        "{}/{}_up{}_order{}.hdf5".format(
            directory, unique_id, args.upsampling[0], args.orders[0]
        ),
        "r",
    )
    ns = jax.random.choice(key, f["a_data"].shape[0], shape=(Np,), replace=False)

    for i, n in enumerate(ns):
        if args.plot_movie:
            anim = movie_data(args, n, T, directory, unique_id, label)
        plot_data(args, n, directory, unique_id, label)
        #compute_entropy(args, n, directory, unique_id, label)
        plt.show()
