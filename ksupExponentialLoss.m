function [w,costs] = ksupExponentialLoss(X,Y,lambda,k,L,w0, ...
                                        iters_acc,eps_acc);
% Author: Matthew Blaschko - matthew.blaschko@inria.fr
% Copyright (c) 2013
%
% Run k-support norm regularized exponential loss
% first 3 arguments are required!
%
% If you use this software in your research, please cite:
%
% M. B. Blaschko, A Note on k-support Norm Regularized Risk Minimization.
% arXiv:1303.6390, 2013.
%
% Argyriou, A., Foygel, R., Srebro, N.: Sparse prediction with the k-support
% norm. NIPS. pp. 1466-1474 (2012)


    if(~exist('eps_acc','var'))
        eps_acc = 1e-4;
    end

    if(~exist('iters_acc','var'))
        iters_acc = 2000; 
    end
    
    
    if(~exist('w0','var'))
        w0 = zeros(size(X,2),1);
    end
    
    if(~exist('k','var'))
        k = round(size(X,2)/4);
    end
    
    if(~exist('L','var'))
        if(size(X,1)>size(X,2)) % lipschitz constant for gradient
            L = eigs(X'*X,1);
        else
            L = eigs(X*X',1);
        end
        L = L*50; % this is unbounded, just choose a large number here.
                  % Try something larger if you suspect you aren't converging 
                  % to correct solution.  Try something smaller if you want to
                  % play fast and loose.
    end
    
    [w,costs] = overlap_nest(@(w)(ExpLoss(w,X,Y)),...
                             @(w)(gradExpLoss(w,X,Y)), lambda, ...
                             L, w0, k, iters_acc,eps_acc);
end

function l = ExpLoss(w,X,Y)
l = sum(exp(-Y.*(X*w)));
end

function g = gradExpLoss(w,X,Y)
g = -X'*(Y.*exp(-Y.*(X*w)));
end

% end of file
