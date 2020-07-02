function [Acc,Cls,Alpha] = ACRLS(Xs,Xt,Ys,Yt,options,K)
%% Mingsheng Long. Adaptation Regularization: A General Framework for Transfer Learning. TKDE 2012.

%% Load algorithm options
addpath(genpath('../liblinear/matlab'));

if nargin < 3
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'p')
    options.p = 5;
end
if ~isfield(options,'sigma')
    options.sigma = 0.1;
end

if ~isfield(options,'gamma')
    options.gamma = 1.0;
end

knn = options.p;
sigma = options.sigma;
gamma = options.gamma;


%% Set predefined variables
X = [Xs;Xt];
Y = [Ys;Yt];
n = size(Xs,1);
m = size(Xt,1);
nm = n+m;
E = diag(sparse([ones(n,1);zeros(m,1)]));
YY = [];
for c = reshape(unique(Y),1,length(unique(Y)))
    YY = [YY,Y==c];
end
[~,Y] = max(YY,[],2);

%% Data normalization
%X = X*diag(sparse(1./sqrt(sum(X.^2))));


%% Construct graph Laplacian
manifold.k = options.p;
manifold.Metric = 'Euclidean';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'HeatKernel';
W = graph1(X,manifold);
Dw = diag(sparse(sqrt(1./sum(W))));
L = speye(nm)-Dw*W*Dw;


Alpha = ((E+gamma*L)*K+sigma*speye(nm,nm))\(E*YY);
F = K*Alpha;
[~,Cls] = max(F,[],2);

%% Compute accuracy

Acc = numel(find(Cls(n+1:end)==Y(n+1:end)))/m;
Cls = Cls(n+1:end);


end
