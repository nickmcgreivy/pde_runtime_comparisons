from arguments import get_args
from batch_train import batch_train_model


def main():
    args = get_args()
    batch_train_model(args, args.train_dir, args.param_dir)


if __name__ == "__main__":
    main()
