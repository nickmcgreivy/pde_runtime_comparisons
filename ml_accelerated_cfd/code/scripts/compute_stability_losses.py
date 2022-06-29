from arguments import get_args
from compute_test_losses import compute_losses
from flux import Flux


def main():
    args = get_args()
    compute_losses(args, args.stability_dir, Flux.LEARNED, "learned", args.inner_loop_steps_stability, mean_loss=False)
    compute_losses(args, args.stability_dir, Flux.CENTERED, "centered", args.inner_loop_steps_stability, mean_loss=False)
    compute_losses(args, args.stability_dir, Flux.UPWIND, "upwind", args.inner_loop_steps_stability, mean_loss=False)


if __name__ == "__main__":
    main()
