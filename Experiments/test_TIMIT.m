addpath(genpath('../'));


%% Load Dataset ----------

% The training set has been generated according to standard practice in speech community,
% as in "Belkin M., & Ma S. (2017). Diving into the shallows: a computational perspective on large-scale shallow learning" (https://arxiv.org/abs/1703.10622),
% see also https://infoscience.epfl.ch/record/192584.
% The testing set is copyrighted so we can not circulate it.
filepath = '/DATASETS/TIMIT.mat';

if ~exist('X' , 'var')
    load(filepath);
end

%% Preprocessing ----------

renorm = @(W, Z) W*(diag(1./(std(Z))));
recenter = @(W, Z) (renorm(W - ones(size(W,1),1)*mean(Z),Z));

ntr = 1124823;
nts = 57242;

Xtr0 = Xtr;
Xtr = recenter(Xtr, Xtr0);
Xtr = Xtr(1:ntr,:);
Xts = recenter(Xts, Xtr0);
Ytr = encode_ys(double(Ytr));
Yts = double(Yts);

clear Xtr0

%% Params ----------

sigma = 15;
kernel = gaussianKernel(sigma);

lambda = 1e-9;

m = 100000;

trp = randperm(ntr,m);
Xuni = Xtr(trp,:);

T = 25;

cobj = [];

callback = @(alpha, cobj) [];

memToUse = [];

useGPU = 1;

%% FALKON ----------

alpha = falkon(Xtr , Xuni , kernel, Ytr, lambda, T, cobj, callback, memToUse, useGPU);

tic; Yp = KtsProd(Xts, Xuni, alpha, 20, kernel); toc

Ypred = decode_ys(Yp);

C_ERR = sum(Ypred ~= Yts)/nts
