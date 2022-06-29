from arguments import get_args
import matplotlib.pyplot as plt
import numpy as np

def main():
	args = get_args()

	fig, axs = plt.subplots(
		len(args.upsampling), len(args.orders), figsize=(12, 7.5), squeeze=False
	)

	for i, order in enumerate(args.orders):
		for j, up in enumerate(args.upsampling):


			with open(
				"{}/{}_order{}_up{}_losses.txt".format(args.param_dir, args.unique_id, order, up), "r"
			) as f:
				losses = f.readlines()
				losses = np.asarray([float(x) for x in losses])

			axs[j,i].plot(losses)

	plt.show()

if __name__ == "__main__":
	main()

