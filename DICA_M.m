function [M,Z,A] = DICA_M(Xs,Xt,Ys,Yt0,K,H,options)

% Load algorithm options
if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'lambda')
    options.lambda = 10.0;
end
if ~isfield(options,'ker')
    options.ker = 'rbf';
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'data')
    options.data = 'default';
end
options.ker = 'linear';
k = options.k;
lambda = options.lambda;
deltha = options.deltha;
% Set predefined variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2)))); % Scale to make columns comparable.
[m, n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
n = ns + nt;
C = length(unique(Ys)); %the number of classes

% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e'*C; % generate M0
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C) % for each class iterate --> sigma(c=1;C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        M = M + e*e'; % M = M0 + Mc
    end
end
% The norm of a matrix is a scalar that gives some measure of the magnitude
% of the elements of the matrix.
M = M/norm(M,'fro'); % The Frobenius-norm of matrix A, sqrt(sum(diag(A'*A))).

% Domain-invariant clustering
Qs = ones(ns,m);
Qt = zeros(nt,m);
XXs = Xs';
%XXt = Xt';
for c = reshape(unique(Ys),1,C) % for each class iterate --> sigma(c=1;C)
    Qs(Ys==c,:) = XXs(Ys==c,:) - repmat(mean(XXs(Ys==c,:)),length(find(Ys==c)),1);
    %Qt(Yt0==c,:) = XXt(Yt0==c,:) - repmat(mean(XXt(Yt0==c,:)),length(find(Yt0==c)),1);
end

Q = [Qs; Qt];
% ****************************

% Transfer One: DICA
G = speye(n);    

    [A,~] = eigs(K*M*K'+ lambda*G+ deltha*Q*Q', K*H*K',k,'SM');
    Z = A'*K;



end