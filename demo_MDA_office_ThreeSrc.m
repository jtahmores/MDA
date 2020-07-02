
clear all;
clc

% Set algorithm parameters
options.k = 20;             % #subspace bases, default=20
options.lambda = 1.0;       % regularizer, default = 1.0
options.deltha = 0.001;       % regularizer, default = 1.0
options.gamma = 1.0;
options.ker = 'linear';     % kernel type, default='linear'
T = 10; % Number of iterations

srcStr = {{'amazon','webcam','dslr'},{'Caltech10','webcam','dslr'},{'Caltech10','amazon','dslr'},{'Caltech10','amazon','webcam'}};
tgtStr = {'Caltech10','amazon','webcam','dslr'};


fileID = fopen(strcat('MDA_Office_ThreeSrc.txt'),'wt');



for iData = 1:4
    
    src = srcStr{iData};
    tgt = char(tgtStr{iData});
    options.data = strcat(src{1},',',src{2},',',src{3},'_vs_',tgt);
    fprintf(fileID,'\n%s\n\n',options.data);
    
    % Preprocess data using Z-score
    load(['data/' src{1} '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs1 = zscore(fts,1);
    Ys1 = labels;
    load(['data/' src{2} '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs2 = zscore(fts,1);
    Ys2 = labels;
    load(['data/' src{3} '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs3 = zscore(fts,1);
    Ys3 = labels;
    Xs=[Xs1;Xs2;Xs3]';
    Ys=[Ys1;Ys2;Ys3];
    load(['data/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xt = zscore(fts,1)';
    Yt = labels;
    
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
    fprintf(fileID,'%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',Acc*100);
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
