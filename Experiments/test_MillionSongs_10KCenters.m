addpath(genpath('../'));


%% Load Dataset ----------

% the dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
% and it needs to be converted from ".txt" to ".mat"
filepath = '/DATASETS/YearPredictionMSD.mat';

if ~exist('X' , 'var')
    load(filepath);
end

%% Preprocessing ----------

renorm = @(W, Z) W*(diag(1./(std(Z))));
recenter = @(W, Z) (renorm(W - ones(size(W,1),1)*mean(Z),Z));

ntr = 463715;
nts = 51630;

Ytr0 = X(1:ntr,1);
Xtr0 = X(1:ntr,2:91);

Xtr = recenter(Xtr0, Xtr0);
Ytr = recenter(Ytr0, Ytr0);
Xts = recenter(X((ntr+1):(ntr+nts),2:91), Xtr0);
Yts = recenter(X((ntr+1):(ntr+nts),1), Ytr0);

clear X Xtr0

%% Params ----------

sigma = 6;
kernel =  gaussianKernel(sigma);

lambda = 1e-6;

m = 10000;

trp =  randperm(ntr,m);
Xuni = Xtr(trp,:);

T = 20;

cobj = [];

callback = @(alpha, cobj) [];

memToUse = [];

useGPU = 1;

%% FALKON ----------

alpha = falkon(Xtr , Xuni , kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU);


tic; Kts = kernel(Xts,Xuni); toc

Ypred = Kts*alpha;

MSE = mean((Ypred*std(Ytr0)-Yts*std(Ytr0)).^2)

UYp = Ypred*std(Ytr0)+mean(Ytr0);
UYts = Yts*std(Ytr0)+mean(Ytr0);
Relative_Error = sqrt(mean(((UYts-UYp)./UYts).^2));
