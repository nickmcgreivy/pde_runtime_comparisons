import sys
import matplotlib.pyplot as plt
import jax.numpy as np
import numpy as onp
import h5py
from jax import vmap
import jax
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from arguments import get_args


def get_linestyle(k):
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    return linestyles[k % len(linestyles)]


def get_color(order):
    colors = ["orange", "blue", "red", "green", "yellow"]
    return colors[order % len(colors)]


def set_labels(fig, axs):
    handles, labels = axs.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    fig.legend(newHandles, newLabels, fontsize=13)


def plot_losses_vs_upsampling(args, dirs, ids, labels):
    fig, axs = plt.subplots(1, 1, figsize=(3, 4))
    for k, unique_id in enumerate(ids):
        for i, order in enumerate(args.orders):
            id_losses = []
            for j, up in enumerate(args.upsampling):
                with open(
                    "{}/{}_up{}_order{}_losses.txt".format(
                        dirs[k], unique_id, up, order
                    ),
                    "r",
                ) as f:
                    losses = f.readlines()
                    losses = [float(x.strip()) for x in losses]
                id_losses.append(np.mean(np.nan_to_num(np.asarray(losses), nan=1e7)))
            axs.loglog(
                args.upsampling,
                id_losses,
                color=get_color(order),
                linewidth=args.linewidth,
                linestyle=get_linestyle(k),
            )
            axs.scatter(
                args.upsampling,
                id_losses,
                color=get_color(order),
                s=5,
            )
            axs.set_xticks(args.upsampling)
            axs.set_xticklabels(args.upsampling)
            axs.minorticks_off()
            axs.set_ylim([1e-4 - 1e-5, 1e0 + 1e-1])

    handles = []
    for order in args.orders:
        handles.append(
            mpatches.Patch(color=get_color(order), label="Order = {}".format(order))
        )
    for k, unique_id in enumerate(ids):
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle=get_linestyle(k),
                linewidth=args.linewidth,
                label=labels[k],
            )
        )
    axs.legend(handles=handles)
    fig.supxlabel("Downsampling factor", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_losses_vs_time(args, dirs, ids, labels):
    fig, axs = plt.subplots(
        1, len(args.upsampling), figsize=(8, 4), sharex=True, sharey=True, squeeze=False
    )
    for k, unique_id in enumerate(ids):
        for i, order in enumerate(args.orders):
            id_losses = []
            for j, up in enumerate(args.upsampling):

                nx = args.nx_max // up
                ny = args.ny_max // up
                dx = args.Lx / (nx)
                dy = args.Ly / (ny)
                dt = args.cfl_safety * ((dx * dy) / (dx + dy)) / (2 * order + 1)

                with open(
                    "{}/{}_up{}_order{}_losses.txt".format(
                        dirs[k], unique_id, up, order
                    ),
                    "r",
                ) as f:
                    losses = f.readlines()
                    losses = np.asarray(
                        [
                            [
                                float(y)
                                for y in x.replace("[", "").replace("]", "").split(", ")
                            ]
                            for x in losses
                        ]
                    )
                loss_averaged = np.mean(np.nan_to_num(losses, nan=1e7), axis=0)

                ts = dt + np.arange(loss_averaged.shape[0]) * dt
                axs[0, j].plot(
                    ts,
                    loss_averaged,
                    color=get_color(order),
                    label="Order = {}".format(order),
                    linewidth=args.linewidth,
                    linestyle=get_linestyle(k),
                )
    for j, up in enumerate(args.upsampling):
        axs[0, j].minorticks_off()
        axs[0, j].set_xlabel(args.upsampling[j])
        # top = 0.2
        _, top = plt.xlim()
        axs[0, j].set_ylim([0.0, top])
        _, top = plt.xlim()
        axs[0, j].set_xlim([0.0, top])

    handles = []
    for order in args.orders:
        handles.append(
            mpatches.Patch(color=get_color(order), label="Order = {}".format(order))
        )
    for k, unique_id in enumerate(ids):
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle=get_linestyle(k),
                linewidth=args.linewidth,
                label=labels[k],
            )
        )
    axs[0, 0].legend(handles=handles, fontsize=8)
    fig.supxlabel("Downsampling factor", fontsize=10)
    fig.suptitle("Sqrt(MSE) vs Time", fontsize=11)
    fig.tight_layout()
    plt.show()
