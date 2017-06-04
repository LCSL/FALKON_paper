# FALKON_paper

Intro
---------------------

This repository provides the code used to run the experiments of the paper "FALKON: An Optimal Large Scale Kernel Method" (https://arxiv.org/abs/1705.10958).
In particular, the folder FALKON contains a preliminary MATLAB implementation of the algorithm, that uses CPU and one GPU.

Installation on LINUX
---------------------

To install the code, set the MATLAB shell path to the FALKON folder and
run:
```matlab
mex -largeArrayDims ./tri_solve_d.cpp -lmwblas -lmwlapack
```
```matlab
mex -largeArrayDims ./inplace_chol.cpp -lmwblas -lmwlapack
```
Furthermore the "free" command line linux tool needs to be installed on your machine.

FALKON
---------------------
The algorithm is implemented by the function
```matlab
alpha = falkon(Xtr, C, kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU)
```
Input:
* `Xtr`, *n* x *d* matrix, containing the training points  (*n* is the number of points and *d* the dimensionality);
* `C`, *m* x *d* matrix, containing the Nystrom centers;
* `kernel`, the kernel to use. For a non-linear kernel it must be a function handler. In particular the function is assumed to take two matrices in input (assume `X1`:`r` x `d` and `X2`:`s`x`d`), and give in output the Gram kernel matrix (of dimension `r`x`s`). E. g. `kernel = @(X1, X2) (1 + X1*X2').^2` for the polynomial kernel of degree two. To specify a linear kernel of the form `q + m *X1*X2` use the notation `{'linear', q, m}`, in this way FALKON will use optimized execution for linear kernels. Finally you can use the helper function `gaussianKernel` and `gaussianKernel_MWs` to produce gaussian kernels, e.g. `kernel = gaussianKernel(5.0)` to gaussian kernel of standard deviation `5.0`.
* `Ytr`,*n* x *t* matrix, containing the labels of the training points  (where *t* is the length of the label vector associated to each point. It is 1 for monovariate regression problems and binary classification, otherwise it is equal to the number of classes, for multiclass classification tasks or for multivariate regression); 
* `lambda`,positive double, the regularization parameter;
* `T`, positive integer, the number of iterations;
* `cobj, callback` respectively a support object and callback function called at the end of each iteration; e.g. `cobj = []; callback = @(alpha, cobj_iter) [];` to do nothing. To understand how to use them, note that at each iteration `cobj_iter = callback(alpha, cobj_iter)`.
* `memToUse`, positive double, the maximum amount of RAM memory to use, in GB; If `memToUse = []` FALKON will compute it automatically.
* `useGPU`, a binary flag to specify if to use the GPU. If the GPU flag is set at 1 the first GPU of the machine will be used.


Output:
* `alpha`, `m`x`t` matrix, containing the vector of coefficients of the model

*Example*:

In this example we assume to have already loaded and preprocessed `Xtr`, `Ytr` (training set) and `Xts`, `Yts` (test set), the following script executes FALKON, for 10 iterations, with a Gaussian kernel of width 15,
a `lambda=0.001`, 10,000 Nystrom centers. Note that in the following code 1) FALKON is not using
any support object and callback, 2) the GPU will be used for the computations and 3) the function will use all the free memory available on the machine (depending on the dimensionality of the problem).

```matlab
addpath(genpath('../'));

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

% empty object
cobj = [];

% empty callback
callback = @(alpha, cobj) [];

% GB of memory to use (using "[]" will allow the machine to use all the free memory)
memToUse = [];

% flag for using or not the GPU
useGPU = 1;

alpha = falkon(Xtr , C , kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU);
```

To test the predictor learned above on the test set `Xts`, `Yts`, we compute the mean square error (MSE) with the help of the function `KtsProd` (that computes `Ypred = kernel(Xts, C)` in blocks), as follows

```matlab

% prediction values on the test set Xts
Ypred = KtsProd(Xts, C, alpha, 10, kernel);

% mean square error
MSE = mean((Yts - Ypred).^2)
```

Experiments
---------------------

The scripts to reproduce the experiments of the paper are in the folder "Experiments".
Each script performs supervised learning on a specific dataset.
In particular the script assumes that the dataset has been already preprocessed and the resulting features saved on disk in ".mat" format, for more details see the comments in the code of each script.

*Example*:

To run FALKON on the MillionSongs dataset with the same setting presented
in the table of the paper (10,000 Nystrom centers):

* Download the dataset (https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD)

* Convert the data format from ".txt" to ".mat" (e.g. see MATLAB `fscanf` function).

* Once you have saved the dataset in the new format, modify the `filepath` variable in the script
`test_MillionSongs_10KCenters.m` with the ".mat" file path.

* Run the script and enjoy the results.
