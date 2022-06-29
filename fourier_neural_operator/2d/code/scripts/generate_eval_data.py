from arguments import get_args
from generate_data import generate_eval_data
from flux import Flux
import jax


def main():
    args = get_args()

    """
    generate_eval_data(
        args,
        args.eval_dir,
        args.num_eval,
        args.evaluation_time,
        Flux.VANLEER,
        "vanleer",
        args.random_seed,
        args.inner_loop_steps,
    )"""
    
    
    generate_eval_data(
        args,
        args.eval_dir,
        args.num_eval,
        args.evaluation_time,
        Flux.UPWIND,
        "upwind",
        args.random_seed,
        args.inner_loop_steps,
    )
    
    
    """
    generate_eval_data(
        args,
        args.eval_dir,
        args.num_eval,
        args.evaluation_time,
        Flux.CENTERED,
        "centered",
        args.random_seed,
        args.inner_loop_steps,
    )
    """
    """
    generate_eval_data(
        args,
        args.eval_dir,
        args.num_eval,
        args.evaluation_time,
        Flux.LEARNED,
        "learned",
        args.random_seed,
        args.inner_loop_steps,
    )
    """
    


if __name__ == "__main__":
    main()
