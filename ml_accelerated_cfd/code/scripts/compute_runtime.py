from arguments import get_args
from generate_data import compute_corrcoef
from flux import Flux

def main():
	args = get_args()

	Tf = 1.0
	orders = [0, 1, 2]
	nxs = [[32, 48, 64, 96, 128, 192, 256], [16, 24, 32, 48, 64, 96, 128], [16, 24, 32, 48, 64, 96, 128]]
	nxs_baseline = [32, 64, 128, 256, 512]
	baseline_dt_reduction = 8.0
	
	compute_runtime(args, orders, nxs, nxs_baseline, baseline_dt_reduction, Tf)




if __name__ == '__main__':
	main()