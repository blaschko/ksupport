function [accTab,mseTab,algNames] = experiments();
% Author: Matthew Blaschko - matthew.blaschko@inria.fr
% Copyright (c) 2013
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
%
% If you use this software in your research, please cite:
%
% M. B. Blaschko, A Note on k-support Norm Regularized Risk Minimization.
% arXiv:1303.6390, 2013.
%
% Argyriou, A., Foygel, R., Srebro, N.: Sparse prediction with the k-support
% norm. NIPS. pp. 1466-1474 (2012)

weights = randn(15,1)*4; % random linear weighting of signal

blaschkoDisp('generating random data');
[Xtrain,Ytrain] = gendata(50,weights);
[Xval,Yval] = gendata(50,weights);
[Xtest,Ytest] = gendata(250,weights);

d = size(Xtest,2);
% set of k values to select from for k-support norm
ks = [1:d];
% set of regularization parameters to select from
lambdas = 10.^[-15:5];

algs = cell(0);
algNames = cell(0);
algs{end+1} = @ksupLeastSquares; %squared loss
algNames{end+1} = 'squared loss';
algs{end+1} = @ksupOneSideLeastSquares; % one sided squared loss
algNames{end+1} = 'one sided squared loss';
algs{end+1} = @ksupSVM; % hinge loss
algNames{end+1} = 'hinge loss';
algs{end+1} = @ksupLogisticRegression; % logistic loss
algNames{end+1} = 'logistic loss';
algs{end+1} = @ksupExponentialLoss; % exponential loss
algNames{end+1} = 'exponential loss';
algs{end+1} = @ksupLeastAbsDeviations; % absolute loss
algNames{end+1} = 'absolute loss';
algs{end+1} = @(X,Y,lambda,k)(ksupEpsilonInsensitive(X,Y,lambda,k,1));% epsilon insensitive loss w/ epsilon=1
algNames{end+1} = 'epsilon-insensitive loss';

accTab = zeros(3,length(algs));
mseTab = zeros(3,length(algs));

for i=1:length(algs)
    % squared loss
    blaschkoDisp(['evaluating ' algNames{i}])
    blaschkoDisp('k-support validation');
    [acc_k,mse_k] = evalMethod(algs{i},Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,ks);
    blaschkoDisp('l1 regularization');
    [acc_1,mse_1] = evalMethod(algs{i},Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,1);
    blaschkoDisp('l2 regularization');
    [acc_2,mse_2] = evalMethod(algs{i},Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,d);
    accTab(:,i) = [acc_k; acc_1; acc_2];
    mseTab(:,i) = [mse_k; mse_1; mse_2];
end

end

function blaschkoDisp(message);
disp([message ' ' datestr(now)]);
end

function [acc,mse,beta] = evalMethod(func,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,ks)

beta = modelSelection(func,Xtrain,Ytrain,Xval,Yval,lambdas,ks);

pred = Xtest*beta;
acc = length(find(sign(pred)==Ytest))/length(Ytest);
mse = norm(Ytest-pred).^2;

end

function beta = modelSelection(func,Xtrain,Ytrain,Xval,Yval,lambdas,ks)

    accval = -Inf;
    beta = zeros(size(Xval,2),1);
    for i=1:length(lambdas)
        for j = 1:length(ks)
            w = func(Xtrain,Ytrain,lambdas(i),ks(j));
            pred = Xval*w;
            acc = length(find(sign(pred)==Yval))/length(Yval);
            if(acc>accval)
                accval = acc;
                beta = w;
            end
        end
    end
end

function [X Y] = gendata(n,weights)

if(~exist('weights','var'))
    weights = [3 -4 2 -3 3 4 1 -0.5 2 1 -3 -1 2 1 -1.5]';
end

Y = sign(rand(n,1)-0.5);

X = [Y*weights' zeros(n,50)];
X = X+randn(size(X))*10;

end
