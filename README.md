# FALKON_paper

Intro
---------------------

In this repository we present you the code used to run the experiments of the paper "FALKON: An Optimal Large Scale Kernel Method" (https://arxiv.org/abs/1705.10958).

Installation on LINUX
---------------------

To install the code move the MATLAB shell path to the FALKON folder and
run:
```
mex -largeArrayDims ./tri_solve_d.cpp -lmwblas -lmwlapack
```
```
mex -largeArrayDims ./inplace_chol.cpp -lmwblas -lmwlapack
```
Furthermore the "free" command line linux tool needs to be installed on your machine.

FALKON
---------------------
The algorithm is implemented by the function
```matlab
falkon(Xtr, C, kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU)
```

which returns the coefficients vector alpha.

The function arguments are respectively: the matrix $\mathcal{R}^{n\times d}$ of training points `Xtr`, the matrix of the Nystrom centers `C`, the kernel to use `kernel`, the labels of the training points `Ytr`, the regularization parameter `lambda`, the number of iterations `T`, a support object `cobj` , a callback function `callback`, the maximum memory to use for the computations `memToUse`, a binary flag indicating if to use the GPU `useGPU`.

Example:

Given `Xtr`, `Ytr` as the training set, the above script executes FALKON with a Guassian kernel of width 15,
a lambda 0.001, 10,000 Nystrom centers for 10 iterations. Note that for how the code has been written, FALKON is not using
any support object and callback. Furthermore the GPU will be used for the computations and
specifying `[]` as `memToUse` the function will use all the free memory available on the machine.

```matlab

% kernel to use
sigma = 15;
kernel = gaussianKernel(sigma);

% regularization parameter lambda
lambda = 1e-3;

% number of Nystrom centers
m = 10000;

% matrix of the Nystrom centers from the training set
ntr = size(Xtr,1);
trp = randperm(ntr,m);
C = Xtr(trp,:);

%number of iterations
T = 10;

%
cobj = [];

callback = @(alpha, cobj) [];

memToUse = [];

useGPU = 1;

alpha = falkon(Xtr , C , kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU);
```

Experiments
---------------------

In the Experiments folder there are scripts to reproduce the experiments of the paper
for the different datasets and with different settings.

Remember to add the correct path of the dataset in these scripts.
