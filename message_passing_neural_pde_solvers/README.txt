Hi,

I have some concerns regarding the paper Message Passing Neural PDE Solvers. 

Concern #1: Regarding E1 and E2, you state that the runtime of the WENO5 baseline on the 1D inviscid Burgers’ equation is as follows:

nx = 100: 1.9s
nx = 50  : 1.8s
nx = 40  : 1.7s

This can’t be right. The step size dt, due to the CFL condition, is proportional to 1/nx. The work per timestep is proportional to nx, so the runtime should go like 1/(nx)^2. To estimate what runtime you might expect from a simple numerical solver, I used a 1D Burgers’ finite volume solver on my laptop CPU. It uses Godunov flux and explicit RK3 timestepping. It does not use WENO5. In my implementation, 80% of the runtime is spent evaluating the sum-of-sines-with-J=5 forcing function. The runtime was as follows:

For E1:

nx = 100: 3.0ms
nx = 50  : 800µs
nx = 40  : 550µs

For E2 (smaller timestep due to diffusion restriction):

nx = 100: 5.4ms
nx = 50  : 1.2ms
nx = 40  : 850µs

Thus, my concern is that your baseline WENO5 solver is about 3 orders of magnitude slower than a simple CPU solver for this equation. 

Concern #2: You write that “All method are implemented for GPU, so runtime comparisons are fair.” However, for WE1, WE2, WE3, the pseudospectral baseline is implemented in scipy on CPU. Clearly, the comparison is not fair. For a fair comparison, the baseline should be implemented in a JIT-compiled library and be evaluated on both CPU and GPU to see which runtime is faster.

Concern #3: Regarding WE1, WE2, WE3, you state that the runtime and losses of the pseudospectral CPU baseline in scipy is as follows:

nx = 100 : 0.60s    loss = 0.004 
nx = 50   : 0.35s    loss = 0.450
nx = 40   : 0.25s    loss = 194.622
nx = 20   : 0.20s    loss = breaks

I ran a different method at low resolution to propagate a gaussian pulse for the wave equation, albeit with periodic boundary conditions and uniform grid spacing. These constraints could be relaxed. The runtime was as follows:

nx = 80  : 5.8ms    loss = near-perfect
nx = 40  : 1.6ms    loss = near-perfect
nx = 20  : 350µs    loss = near-perfect

I don’t compute the loss, but you can see the near-perfect accuracy at nx=80,40,20 in the video below.

Thus, I have two concerns regarding your baseline solver. First, it is about 2 to 3 orders of magnitude slower than an advanced numerical method at the same resolution. Second, it is not fair to use a baseline method that breaks, especially when methods exist to solve this equation with near-perfect accuracy at the same resolution.


I look forward to seeing the updated paper.

Thanks,
Nick McGreivy