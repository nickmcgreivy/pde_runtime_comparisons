import matplotlib.pyplot as plt
import jax
import h5py
from arguments import get_args
from plot_data import analyze_data


def main():
    args = get_args()
    dirs = [args.eval_dir, args.eval_dir, args.eval_dir]
    #ids = ["learned", "centered", "upwind"]
    #labels = ["MLR", "centered", "upwind"]
    ids = ["conservation"]
    labels = ["conservation"]
    for i, unique_id in enumerate(ids):
        analyze_data(
            args, args.num_eval_plot, args.evaluation_time, dirs[i], unique_id, labels[i]
        )


if __name__ == "__main__":
    main()
