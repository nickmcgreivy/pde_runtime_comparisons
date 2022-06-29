from arguments import get_args
from generate_data import print_runtime
from flux import Flux

def main():
	args = get_args()
	inner_loop_steps = 1

	
	#########
	"""
	print("printing exact runtime")
	order = 2
	nx = ny = 2
	print_runtime(args, args.evaluation_time, 
		Flux.UPWIND, "None", args.random_seed, 
		inner_loop_steps, 
		nx, 
		ny, 
		order)
	"""


	#########
	order = 0
	nxs = [32, 64, 128]

	for nx in nxs:
		ny = nx
		print("Flux is VANLEER")
		print_runtime(args, args.evaluation_time, 
			Flux.VANLEER, "None", args.random_seed, 
			inner_loop_steps, 
			nx, 
			ny, 
			order)


		print("Flux is ARTIFICIAL STABILITY")
		print_runtime(args, args.evaluation_time, 
			Flux.CONSERVATION, "None", args.random_seed, 
			inner_loop_steps, 
			nx, 
			ny, 
			order)


	"""
	#########
	order = 1
	nxs = [24, 48, 96]

	for nx in nxs:
		ny = nx
		print_runtime(args, args.evaluation_time, 
			Flux.UPWIND, "None", args.random_seed, 
			inner_loop_steps, 
			nx, 
			ny, 
			order)



	#########
	order = 2
	nxs = [24, 48, 96]

	for nx in nxs:
		ny = nx
		print_runtime(args, args.evaluation_time, 
			Flux.UPWIND, "None", args.random_seed, 
			inner_loop_steps, 
			nx, 
			ny, 
			order)
	"""
	

if __name__ == '__main__':
	main()