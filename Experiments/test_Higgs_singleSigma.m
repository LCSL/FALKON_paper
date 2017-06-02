addpath(genpath('../'));


%% Load Dataset ----------

% the dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/HIGGS
% and it needs to be converted from ".csv" to ".mat"
filepath = '/DATASETS/Higgs.mat';

if ~exist('X' , 'var')
    load(filepath);
end

%% Preprocessing ----------

renorm = @(W, Z) W*(diag(1./(std(Z))).^2);
recenter = @(W, Z) (renorm(W - ones(size(W,1),1)*mean(Z),Z));

[n,d] = size(X);
ntr = ceil(n*.8);
nts = n-ntr;

idx_resh =randperm(n,n);
X = X(idx_resh,:);

Xtr0 = X(1:ntr,2:29);
Xtr = recenter(Xtr0, Xtr0);
Ytr = X(1:ntr,1) .*2 - 1;
Xts = recenter(X((ntr+1):(ntr+nts),2:29), Xtr0);
Yts = X((ntr+1):(ntr+nts),1) .*2 - 1;

clear X Xtr0

%% Params ----------

sigma = 5;
kernel = gaussianKernel(sigma);

m = 100000;

trp = randperm(ntr,m);
Xuni = Xtr(trp,:);

lambda = 1e-8;

T = 20;

cobj = [];

callback = @(alpha, cobj) [];

memToUse = [];

useGPU = 1;

%% FALKON ----------

alpha = falkon(Xtr , Xuni , kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU);

tic; Ypred = KtsProd(Xts, Xuni, alpha, 50, kernel); toc

MSE = mean((Yts-Ypred).^2)
[~,~,~,AUC] = perfcurve(Yts,Ypred,1)
