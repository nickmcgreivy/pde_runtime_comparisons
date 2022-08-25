Need to download:

pybind11
jax
sympy
h5py

cd code/generate_sparse_solve
tar -xvf eigen-3.4.0.tar.bz2 
make compilemac
mv custom_call_* ..
cd ../..


Need to mention how the reduction in timestep of the ML-accelerated CFD was chosen. This is about ensuring the baseline we are comparing to is a strong baseline.