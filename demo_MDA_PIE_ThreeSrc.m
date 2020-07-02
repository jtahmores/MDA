
clear all;
clc

% Set algorithm parameters
options.k = 60;             % #subspace bases, default=20
options.lambda = 0.1;       % regularizer, default = 1.0
options.deltha = 0.1;       % regularizer, default = 1.0
options.gamma = 1.0;
options.ker = 'linear';     % kernel type, default='linear'
T = 10; % Number of iterations

srcStr = {{'PIE05','PIE07','PIE09'},{'PIE05','PIE07','PIE09'},{'PIE05','PIE07','PIE27'},{'PIE05','PIE07','PIE27'},{'PIE05','PIE07','PIE29'},{'PIE05','PIE07','PIE29'},{'PIE05','PIE09','PIE27'},{'PIE05','PIE09','PIE27'},{'PIE05','PIE09','PIE29'},{'PIE05','PIE09','PIE29'},{'PIE05','PIE27','PIE29'},{'PIE05','PIE27','PIE29'},{'PIE07','PIE09','PIE27'},{'PIE07','PIE09','PIE27'},{'PIE07','PIE09','PIE29'},{'PIE07','PIE09','PIE29'},{'PIE07','PIE27','PIE29'},{'PIE07','PIE27','PIE29'},{'PIE09','PIE27','PIE29'},{'PIE09','PIE27','PIE29'}};
tgtStr = {'PIE27','PIE29','PIE09','PIE29','PIE09','PIE27','PIE07','PIE29','PIE07','PIE27','PIE07','PIE09','PIE05','PIE29','PIE05','PIE27','PIE05','PIE09','PIE05','PIE07'};


fileID = fopen(strcat('MDA_PIE_ThreeSrc.txt'),'wt');



for iData = 1:20
    src = srcStr{iData};
    tgt = char(tgtStr{iData});
    options.data = strcat(src{1},',',src{2},',',src{3},'_vs_',tgt);
    fprintf(fileID,'\n%s\n\n',options.data);
    
    
    % Preprocess data using L2-norm
    load(['data/' src{1}]);
    fea = fea ./ repmat(sum(fea,2),1,size(fea,2));
    Xs1 = zscore(fea,1);
    Ys1 = gnd;
    load(['data/' src{2}]);
    fea = fea ./ repmat(sum(fea,2),1,size(fea,2));
    Xs2 = zscore(fea,1);
    Ys2 = gnd;
    load(['data/' src{3}]);
    fea = fea ./ repmat(sum(fea,2),1,size(fea,2));
    Xs3 = zscore(fea,1);
    Ys3 = gnd;
    Xs=[Xs1;Xs2;Xs3]';
    Ys=[Ys1;Ys2;Ys3];
    load(['data/' tgt]);
    fea = fea ./ repmat(sum(fea,2),1,size(fea,2));
    Xt = zscore(fea,1)';
    Yt = gnd;
    
    clear X_src X_tar Y_src Y_tar
    
    X = [Xs,Xt];
    X = X*diag(sparse(1./sqrt(sum(X.^2)))); % Scale to make columns comparable.
    ns = size(Xs,2);
    nt = size(Xt,2);
    n = ns + nt;
    C = length(unique(Ys)); %the number of classes
    e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    M = e*e'*C; % generate M0
    % Construct kernel matrix
    K = kernel(options.ker,X,[],options.gamma);
    % Construct centering matrix
    H = eye(n)-1/(n)*ones(n,n);
    Cls=[];
    G = speye(n);
    Acc = [];
    Acc1 = [];
    Acc2 = [];
    fprintf('DICA:  data=%s  k=%d  lambda=%f  deltha=%f\n',options.data,options.k,options.lambda,options.deltha);
    for i=1:T
        [M,Z,A] = DICA_M(Xs,Xt,Ys,Cls,K,H,options);
        
        knn_model = fitcknn(Z(:,1:ns)',Ys,'NumNeighbors',1);
        Cls = knn_model.predict(Z(:,ns+1:end)');
        
        acc = sum(Cls==Yt)/nt;
        Acc = [Acc;acc(1)];
        fprintf('[%d]  acc=%f\n',i,full(acc(1))*100);
    end
    MM=M;
    AA=A;
    for i=1:T
        [GG,Z,A] = DICA_G(Xs,Xt,Ys,MM,AA,K,H,G,options);
        G=GG;
        
        knn_model = fitcknn(Z(:,1:ns)',Ys,'NumNeighbors',1);
        Cls = knn_model.predict(Z(:,ns+1:end)');
        
        acc = sum(Cls==Yt)/nt;
        Acc = [Acc;acc(1)];
        fprintf('[%d]  acc=%f\n',i,full(acc(1))*100);
    end
    fprintf('Algorithm DICA terminated!!!\n');
    fprintf(fileID,'%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',Acc*100);
    
    %**********************************************************************************************
    %Adaptive classifier
    %**********************************************************************************************
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    options.ker = 'rbf';
    K=kernel(options.ker,Z,[],options.gamma);
    ZZ=Z';
    op.p = 5;             % keep default
    [acc2,Cls2,Alpha] = ACRLS(ZZ(1:ns,:),ZZ(ns+1:end,:),Ys,Yt,op,K);
    fprintf(fileID,'accuracy for ADICA: %.2f \n',acc2 *100);
    fprintf('accuracy for ADICA: %0.2f \n',acc2*100 );
    
end

fclose(fileID);
