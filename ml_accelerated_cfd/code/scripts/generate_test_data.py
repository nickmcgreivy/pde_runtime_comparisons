from arguments import get_args
from generate_data import generate_test_data


def main():
    args = get_args()
    generate_test_data(
        args, args.test_dir, args.num_test, args.testing_time, args.random_seed + 1, args.inner_loop_steps,
    )


if __name__ == "__main__":
    main()
