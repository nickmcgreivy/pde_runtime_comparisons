from arguments import get_args
from generate_data import compute_corrcoef
from flux import Flux

def main():
	args = get_args()

	Tf = 40.0
	Np = 40

	#orders = [0, 1, 2]
	#nxs = [[16, 24, 32, 48, 64, 96, 128, 192, 256], [16, 24, 32, 48, 64, 96, 128], [16, 24, 32, 48, 64, 96]]
	orders = [0, 1, 2]
	nxs = [[32, 64, 128], [16, 32, 64], [16, 32]]
	compute_corrcoef(args, orders, nxs, Tf, Np)




if __name__ == '__main__':
	main()