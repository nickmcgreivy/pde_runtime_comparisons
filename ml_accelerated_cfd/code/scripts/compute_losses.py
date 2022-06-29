from arguments import get_args
from compute_test_losses import compute_losses
from flux import Flux


def main():
    args = get_args()
    compute_losses(args, args.test_dir, Flux.LEARNED, args.inner_loop_steps, "learned")
    compute_losses(args, args.test_dir, Flux.CENTERED, args.inner_loop_steps, "centered")
    compute_losses(args, args.test_dir, Flux.UPWIND, args.inner_loop_steps, "upwind")


if __name__ == "__main__":
    main()
