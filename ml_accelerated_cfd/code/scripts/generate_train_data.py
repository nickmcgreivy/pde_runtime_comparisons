from arguments import get_args
from generate_data import generate_train_data


def main():
    args = get_args()
    generate_train_data(
        args, args.train_dir, args.num_train, args.training_time, args.random_seed,
    )


if __name__ == "__main__":
    main()
