from arguments import get_args
from plot_data import analyze_data


def main():
    args = get_args()
    dirs = [args.test_dir]
    ids = [args.unique_id]
    labels = ["test data"]
    for i, unique_id in enumerate(ids):
        analyze_data(
            args, args.num_test_plot, args.testing_time, dirs[i], unique_id, labels[i]
        )


if __name__ == "__main__":
    main()
