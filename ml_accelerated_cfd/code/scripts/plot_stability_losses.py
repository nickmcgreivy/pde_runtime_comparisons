from arguments import get_args
from compare_models import plot_losses_vs_time as plot_losses


def main():
    args = get_args()
    dirs = [args.stability_dir, args.stability_dir, args.stability_dir]
    ids = ["learned", "centered", "upwind"]
    labels = ["MLR", "Centered", "Upwind"]
    plot_losses(args, dirs, ids, labels)


if __name__ == "__main__":
    main()
