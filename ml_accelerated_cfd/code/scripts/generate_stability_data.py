from arguments import get_args
from generate_data import generate_eval_data, generate_exact_data
from flux import Flux


def main():
    args = get_args()
    
    
    generate_test_data(
        args,
        args.stability_dir,
        args.num_test_stability,
        args.stability_time,
        args.random_seed + 2,
        args.inner_loop_steps_stability,
    )



    generate_eval_data(
        args,
        args.stability_dir,
        args.num_test_stability,
        args.stability_time,
        Flux.LEARNED,
        "learned",
        args.random_seed + 2,
        args.inner_loop_steps_stability,
    )

    generate_eval_data(
        args,
        args.stability_dir,
        args.num_test_stability,
        args.stability_time,
        Flux.CENTERED,
        "centered",
        args.random_seed + 2,
        args.inner_loop_steps_stability,
    )

    generate_eval_data(
        args,
        args.stability_dir,
        args.num_test_stability,
        args.stability_time,
        Flux.UPWIND,
        "upwind",
        args.random_seed + 2,
        args.inner_loop_steps_stability,
    )


if __name__ == "__main__":
    main()
