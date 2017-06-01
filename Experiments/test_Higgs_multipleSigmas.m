addpath(genpath('../'));

%% Load Dataset ----------

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

sigmas = [-2.969724653678234
  -2.104455363627951
  -3.268928214293932
  -2.151117366377688
  -2.808103508980782
  -3.376037554712100
  -2.919790466486163
  -3.021059170715441
  -2.320239620049103
  -3.888428982502075
  -2.257087209313980
  -3.000920557112881
  -2.657348305291945
  -4.268044515534893
  -3.004266130366929
  -2.752586668507473
  -1.308494558067407
  -4.626082630862038
  -2.624656678426839
  -2.651597002422721
  -1.467819569891172
  -3.604263204937710
  -4.047027962696840
  -9.132378213404181
  -3.464485827859968
  -0.183304014560211
  -2.902452018286060
  -3.508393560237325];
S = diag(exp(sigmas));
kern = gaussianKernel_MWs(S);

m = 100000;

trp = randperm(ntr,m);
Xuni = Xtr(trp,:);

lambda = 1e-8;

T = 20;

callback = @(alpha, cobj) cobj;

%% FALKON ----------

alpha = falkon(Xtr , Xuni , kern, Ytr, lambda, T, [], callback, [], 1);

tic; Ypred = KtsProd(Xts, Xuni, alpha, 50, kern); toc

MSE = mean((Yts-Ypred).^2)
[~,~,~,AUC] = perfcurve(Yts,Ypred,1)
