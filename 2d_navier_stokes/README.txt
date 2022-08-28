Need to download:

pybind11
jax
jaxlib
sympy
h5py
tree_math
torch
matplotlib
seaborn (maybe?)
jax-cfd
xarray (maybe?)

cd code/generate_sparse_solve
tar -xvf eigen-3.4.0.tar.bz2 
make compilemac
mv custom_call_* ..
cd ../..


Need to mention how the reduction in timestep of the ML-accelerated CFD was chosen. Same with ML-accelerated CFD. This is about ensuring the baseline we are comparing to is a strong baseline.