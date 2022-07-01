from arguments import get_args
from generate_data import compute_corrcoef
from flux import Flux

def main():
	args = get_args()

	Tf = 10.0
	Np = 100
	orders = [0, 1, 2]
	nxs = [[32, 48, 64, 96, 128, 192, 256], [16, 24, 32, 48, 64, 96, 128], [16, 24, 32, 48, 64, 96]]
	baseline_dt_reductions = [4.0, 6.0, 8.0]
	nxs_baseline = [32, 64, 128, 256, 512]
	
	compute_corrcoef(args, orders, nxs, nxs_baseline, baseline_dt_reductions, Tf, Np)




if __name__ == '__main__':
	main()