# FALKON_paper

Intro
---------------------

In this repository we present you the code used to run the experiments of the paper "FALKON: An Optimal Large Scale Kernel Method" (https://arxiv.org/abs/1705.10958).

Installation on LINUX
---------------------

To install the code move the MATLAB shell path to the FALKON_TOOLS folder and
run:

mex -largeArrayDims ./tri_solve_d.cpp -lmwblas -lmwlapack

mex -largeArrayDims ./inplace_chol.cpp -lmwblas -lmwlapack

Furthermore the "free" command line linux tool needs to be installed on your machine.

FALKON
---------------------

The algorithm is implemented by the function

falkon(X, C, kernel, y, lambda, T, cbobj, callback, memToUse, useGPU)

which returns the coefficients vector alpha.
The function arguments are respectively: the training points "X", the matrix of the Nystrom centers "C", the kernel to use, the labels of the training points "y", the regularization parameter "lambda", the number of iterations "T", a support function "cobj" (optional), a callback function (optional), the maximum memory to use for the computations, a binary flag indicating if to use the GPU.

Experiments
---------------------

In the Experiments folder there are scripts to reproduce the experiments of the paper
for the different datasets and with different settings.

Remember to add the correct path of the dataset in these scripts.
