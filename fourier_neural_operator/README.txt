I am interested in evaluating the accuracy of two statements regarding this paper:

1. "The Fourier neural operator is the first ML-based method to successfully model turbulent flows with zero-shot super-resolution." 

2. "It is up to three orders of magnitude faster compared to traditional PDE solvers."


They use the 2D incompressible Navier-Stokes equation (i.e., incompressible euer equation) with nu = 1e-3, 1e-4, 1e-5, x,y = [0,1], t = [0, T], and forcing function f(x,y) = 0.1 * (sin( 2 pi (x + y)  ) + cos( 2 pi (x + y) )




My DG p=2 implementation runs at 8x8 resolution with 1% accuracy with nu=1e-3 and ~10% accuracy with nu=1e-4 in 0.1s on CPU, while their pseudo-spectral implementation runs in 2.2s on GPU. So I've beat them by a factor of 22, even without implementing on GPU.

Still more work to do to on this paper.