addpath(genpath('./'));
addpath(genpath('../FALKON/'));
addpath(genpath('./Utilities'));

%% Load Dataset ----------

filepath = '/DATASETS/YELP.mat';

if ~exist('X' , 'var')
    load(filepath);
end

%% Preprocessing ----------

[n,d] = size(X);
ntr = ceil(n*.8);
nts = n - ntr;

idx_resh = randperm(n,n);
X = X(idx_resh,:);
Y = Y(idx_resh);

Xtr = X(1:ntr,:);
Ytr = Y(1:ntr,1);
Xts = X(ntr+1:nts+ntr,:);
Yts = Y(ntr+1:nts+ntr,1);

clear X Y;

%% Params ----------

m = 100000;

trp = randperm(ntr,m);

colstr = sum(Xtr,1) > 0;
colsts =sum(Xts,1) > 0;
coltot = (colstr | colsts);
Xuni = Xtr(trp, coltot);
Xtr = Xtr(:, coltot);
Xts = Xts(:, coltot);
coluni = sum(Xuni,1) > 0;
Xtr = Xtr(:, coluni);
Xts = Xts(:, coluni);
Xuni = Xuni(:,coluni);

sigma = 31.4;
beta = 1;
kernel = struct('func', 'linear', 'param1', beta, 'param2', sigma);

lambda = 1e-7;

T = 35;

callback = @(alpha, cobj) cobj;

%% FALKON ----------

alpha = falkon(Xtr, Xuni, kernel, Ytr, lambda, T, [], callback, [], 1);

tic; Ypred = KtsProd(Xts, Xuni, alpha, 10, kernel); toc

RMSE = sqrt(mean((Yts - Ypred).^2))
