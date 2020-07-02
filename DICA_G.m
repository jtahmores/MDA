function [GG,Z,A] = DICA_G(Xs,Xt,Ys,M,AA,K,H,G,options)

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
ker = options.ker;
gamma = options.gamma;
deltha = options.deltha;
% Set predefined variables
A=AA;
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2)))); % Scale to make columns comparable.
[m, n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
n = ns + nt;
C = length(unique(Ys)); %the number of classes


% Transfer One: DICA

    [A,~] = eigs(K*M*K'+lambda*G, K*H*K',k,'SM');
    
    % update G
    G(1:ns,1:ns) = diag(sparse(1./(sqrt(sum(A(1:ns,:).^2,2)+eps))));
    GG=G;
    Z = A'*K;




end