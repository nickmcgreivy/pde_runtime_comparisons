from generate_data import compute_error
from arguments import get_args
import jax
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

def main():
	args = get_args()


	nxs = [8, 16, 32]
	cfls = [6.0, 4.0, 4.0]
	Tf = args.evaluation_time
	Np = int(Tf)
	key = jax.random.PRNGKey(args.random_seed)


	for _ in range(10):
		key, subkey = jax.random.split(key)
		errors = compute_error(args, nxs, cfls, Tf, Np, subkey)

		print(np.mean(errors, axis=-1))


if __name__ == '__main__':
	main()