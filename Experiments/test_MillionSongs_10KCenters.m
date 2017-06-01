addpath(genpath('../'));


%% Load Dataset ----------

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
kern =  gaussianKernel(sigma);

lambda = 1e-6;

m = 10000;

trp =  randperm(ntr,m);
Xuni = Xtr(trp,:);

T = 20;

callback = @(alpha, cobj) cobj;

%% FALKON ----------

alpha = falkon(Xtr , Xuni , kern, Ytr, lambda, T, [], callback, [], 1);


tic; Kts = kern(Xts,Xuni); toc

Ypred = Kts*alpha;

MSE = mean((Ypred*std(Ytr0)-Yts*std(Ytr0)).^2)

UYp = Ypred*std(Ytr0)+mean(Ytr0);
UYts = Yts*std(Ytr0)+mean(Ytr0);
Relative_Error = sqrt(mean(((UYts-UYp)./UYts).^2));
