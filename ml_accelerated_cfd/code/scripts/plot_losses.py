from arguments import get_args
from compare_models import plot_losses_vs_upsampling as plot_losses


def main():
    args = get_args()
    dirs = [args.test_dir, args.test_dir, args.test_dir]
    ids = ["learned", "centered", "upwind"]
    labels = ["MLR", "Centered", "Upwind"]
    plot_losses(args, dirs, ids, labels)


if __name__ == "__main__":
    main()
