I am interested in evaluating the accuracy of two statements regarding this paper:

1. "The Fourier neural operator is the first ML-based method to successfully model turbulent flows with zero-shot super-resolution." 

This appears to be accurate. The flows do appear to be turbulent, as far as I can tell. 

2. "It is up to three orders of magnitude faster compared to traditional PDE solvers."

The DG code is ~16x slower than the 5ms ML code, not 3 orders of magnitude slower.

They use the 2D incompressible Navier-Stokes equation (i.e., incompressible euer equation) with nu = 1e-3, 1e-4, 1e-5, x,y = [0,1], t = [0, T], and forcing function f(x,y) = 0.1 * (sin( 2 pi (x + y)  ) + cos( 2 pi (x + y) )

I've now implemented their exact initial condition, using the code from their Github.


Test 1

nu = 1e-4
cfl_safety = 6.0
args.evaluation_time = 30.0
nx = ny = 8
order = 2

Max Value of Vorticity: 8-10
Runtime: 0.085 (Laptop CPU)
Error: 8.1% (averaged over 10 initializations, variance ~ 2%)

Test 2

nu = 1e-3
cfl_safety = 11.0
args.evaluation_time = 50.0
nx = ny = 8
order = 2

Max Value of Vorticity: 3.0
Runtime: 0.075 (Laptop CPU)
Error: 0.5% (averaged over 10 initializations, variance ~ 0.2%)


My DG p=2 implementation runs at 8x8 resolution with 0.5% accuracy with nu=1e-3 and ~8% accuracy with nu=1e-4 in 0.08s on CPU, while their pseudo-spectral implementation runs in 2.2s on GPU. 