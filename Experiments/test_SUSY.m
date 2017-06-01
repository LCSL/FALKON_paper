addpath(genpath('../'));


%% Load Dataset ----------

% the dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/SUSY
% and it needs to be converted from ".csv" to ".mat"
filepath = '/DATASETS/Susy.mat';

if ~exist('X' , 'var')
    load(filepath);
end

%% Preprocessing ----------

renorm = @(W, Z) W*(diag(1./(std(Z))));
recenter = @(W, Z) (renorm(W - ones(size(W,1),1)*mean(Z),Z));

[n,d] = size(X);
ntr = ceil(n*.8);
nts = n-ntr;

idx_resh =randperm(n,n);
X = X(idx_resh,:);

Xtr0 = X(1:ntr,2:end);
Xtr = recenter(Xtr0, Xtr0);
Ytr = X(1:ntr,1) .*2 - 1;
Ytr(Ytr == 1) = 1;
Ytr(Ytr == -1) = -1;
Xts = recenter(X((ntr+1):(ntr+nts),2:end),Xtr0);
Yts = X((ntr+1):(ntr+nts),1) .*2 - 1;

clear X Xtr0

%% Params ----------

sigma = 4;
kern =gaussianKernel(sigma);

m = 10000;

trp = randperm(ntr,m);
Xuni = Xtr(trp,:);

lambda = 1e-6;

callback = @(alpha, cobj) cobj;

T = 20;

%% FALKON ----------

alpha = falkon(Xtr , Xuni , kern, Ytr, lambda, T, [], callback, [], 1);

tic; Ypred = KtsProd(Xts, Xuni, alpha, 20, kern); toc

MSE = mean((Yts-Ypred).^2)
C_ERR = mean(Yts ~= sign(Ypred))
[~,~,~,AUC] = perfcurve(Yts,Ypred,1)
