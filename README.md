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

The function arguments are respectively: the matrix *n* x *d* of training points `Xtr` (where *n* is the number of points and *d* the dimensionality), the matrix *m* x *d* of *m* Nystrom centers `C`, the kernel to use `kernel`, the matrix *n* x *t* of the labels of the training points `Ytr` (where *t* is 1 for regression problems and binary classification ones, otherwise for multiclass classification tasks it is the number of classes), the regularization parameter `lambda`, the number of iterations `T`, a support object `cobj` , a callback function `callback`, the maximum GB of memory to use for the computations `memToUse`, a binary flag indicating if to use the GPU `useGPU`.

If the GPU flag is set at 1 the first GPU of the machine will be used.

*Example*:

Assuming to have already loaded and preprocessed `Xtr`, `Ytr` as the training set and `Xts`, `Yts` as the test set, the following script executes FALKON with a Guassian kernel of width 15,
a lambda 0.001, 10,000 Nystrom centers for 10 iterations. Note that for how the code has been written, FALKON is not using
any support object and callback. Furthermore the GPU will be used for the computations and
specifying `[]` as `memToUse` the function will use all the free memory available on the machine.

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

To test the predictor learned above on the test set `Xts`, `Yts`, we compute the mean square error (MSE) with the help of the function `KtsProd` as follows

```matlab

% prediction values on the test set Xts
Ypred = KtsProd(Xts, C, alpha, 10, kernel);

% mean square error
MSE = mean((Yts - Ypred).^2)
```

Experiments
---------------------

In the Experiments folder there are scripts to reproduce the experiments of the paper
for the different datasets and with different settings.

These scripts assume to receive the respective dataset with the features already extracted
where specified (see comments in the code) and in a ".mat" format.

*Example*:

To run FALKON on the MillionSongs dataset with the same setting presented
in the table of the paper (10,000 Nystrom centers):

* Download the dataset (https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD)

* Convert the data format from ".txt" to ".mat" (e.g. see MATLAB `fscanf` function).

* Once you have saved the dataset in the new format, modify the `filepath` variable in the script
`test_MillionSongs_10KCenters.m` with the ".mat" file path.

* Run the script and enjoy the results.
