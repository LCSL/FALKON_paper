# FALKON_paper

In this repository we present the code to reproduce the results in the experiment section of the paper "FALKON: An Optimal Large Scale Kernel Method".


Navigate in the matlab shell to the path containing FALKON's code

Run in the matlab shell:

mex -largeArrayDims ./tri_solve_d.cpp -lmwblas -lmwlapack

mex -largeArrayDims ./inplace_chol.cpp -lmwblas -lmwlapack
