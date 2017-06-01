addpath(genpath('../'));


%% Load Dataset ----------

% The training set has been generated extracting the 3-grams from the plain text of the reviews of the "YELP Dataset round 6" as 
% "Roelofs R., Recht B., Tu S., & Venkataraman S. (2016). Large Scale Kernel Learning using Block Coordinate Descent" (https://arxiv.org/abs/1602.05310)
% Then for each training point (representing a review) has been saved as a binary sparse vector that indicate if a 3-gram is present in the review.
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

m = 50000;

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
