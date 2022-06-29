import argparse
import jax.numpy as np
from flux import Flux

PI = np.pi


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_eval",
        help="",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_eval_plot",
        help="",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--equation",
        help="advection, euler, or hw",
        default="advection",
        type=str,
    )
    parser.add_argument(
        "--evaluation_time",
        help="",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--train_dir",
        help="Where the training data is stored",
        default="output",
        type=str,
    )
    parser.add_argument(
        "--poisson_dir",
        help="",
        default="",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        help="Where the test data is stored",
        default="output",
        type=str,
    )
    parser.add_argument(
        "--eval_dir",
        help="Where the evaluation data is stored",
        default="output",
        type=str,
    )
    parser.add_argument(
        "--stability_dir",
        help="Where the test data for extended time evaluation (stability) is stored",
        default="output",
        type=str,
    )
    parser.add_argument(
        "--read_write_dir",
        help="",
        default="",
        type=str,
    )
    parser.add_argument(
        "--param_dir",
        help="Where the parameters are stored as a result of the training",
        default="output",
        type=str,
    )
    parser.add_argument(
        "--nx_max",
        help="",
        default=16 * 8,
        type=int,
    )
    parser.add_argument(
        "--ny_max",
        help="",
        default=16 * 8,
        type=int,
    )
    parser.add_argument(
        "--order_max",
        help="order used for data generation",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--random_seed",
        help="",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--upsampling",
        help="nx = args.nx/args.upsampling",
        default=[4, 8, 16],
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--orders",
        default=[1, 2],
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "-id",
        "--unique_id",
        help="",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--Lx",
        help="",
        default=2 * PI,
        type=float,
    )
    parser.add_argument(
        "--Ly",
        help="",
        default=2 * PI,
        type=float,
    )
    parser.add_argument(
        "--diffusion_coefficient",
        help="",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--forcing_coefficient",
        help="",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--damping_coefficient",
        help="",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--cfl_safety",
        help="",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--exact_flux",
        help="Flux class",
        choices=list(Flux),
        type=Flux,
        default=Flux.UPWIND,
    )
    parser.add_argument(
        "--runge_kutta",
        help="What RK time-stepper am I using?",
        type=str,
        default="rk3",
    )
    parser.add_argument(
        "--max_k",
        help="Maximum wavenumber of initialization",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--min_k",
        help="Minimum wavenumber of initialization",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_init_modes",
        help="",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--amplitude_max",
        help="",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--advection_function",
        help="What is the prescribed phi for 2d advection?",
        type=str,
        default="zero",
    )
    parser.add_argument(
        "--is_forcing",
        help="Am I using Kolmogorov forcing?",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--labels",
        help="List of the labels for each unique ID",
        default=["test label"],
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--linewidth",
        help="",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--plot_movie",
        help="Do I plot a movie?",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--frames_per_time",
        help="Movie frames per unit time",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--movie_delay",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--percent",
        help="Do I plot the percentage change?",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--initial_condition",
        help="What initial condition am I using?",
        default="zero",
        type=str,
    )
    parser.add_argument(
        "--burn_in_time",
        help="",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--inner_loop_steps",
        help="",
        default=10,
        type=int,
    )
    
    if argv is not None:
        return parser.parse_args(argv)
    else:
        return parser.parse_args()
