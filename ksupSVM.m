function [w,costs] = ksupSVM(X,Y,lambda,k,w0,h, ...
                                        iters_acc,eps_acc);
% Author: Matthew Blaschko - matthew.blaschko@inria.fr
% Copyright (c) 2012-2013
%
% Run Ksupport norm using hinge loss function
% first 3 arguments are required!
%
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
%
% If you use this software in your research, please cite:
%
% M. B. Blaschko, A Note on k-support Norm Regularized Risk Minimization.
% arXiv:1303.6390, 2013.
%
% Argyriou, A., Foygel, R., Srebro, N.: Sparse prediction with the k-support
% norm. NIPS. pp. 1466-1474 (2012)

    if(nargin<8)
        eps_acc = 1e-4;
    end

    if(nargin<7)
        iters_acc = 2000; 
    end
    
    if(nargin<6) % Huber parameter (see e.g. Olivier
                 % Chapelle. Training a Support Vector Machine in
                 % the Primal, Neural Computation, 2007. Eq. (18))
        h = 0.1;
    end
    
    if(nargin<5)
        w0 = zeros(size(X,2),1);
    end
    
    if(nargin<4)
        k = round(size(X,2)/4);
    end
           
    if(size(X,1)>size(X,2)) % lipschitz constant for gradient of squared loss
        L = eigs(X'*X,1)/(2*h);
    else
        L = eigs(X*X',1)/(2*h);
    end
    [w,costs] = overlap_nest(@(w)(huberLoss(w,X,Y,h)),...
                             @(w)(gradHuberLoss(w,X,Y,h)), lambda, ...
                             L, w0, k, iters_acc,eps_acc);
end

function [ind1,ind2] = huberInd(w,X,Y,h);
  margin = Y.*(X*w);
  ind1 = find(margin<1-h);
  ind2 = find(abs(1-margin)<=h);
end

function l = huberLoss(w,X,Y,h)
    [ind1,ind2] = huberInd(w,X,Y,h);
    l = 0;
    if(length(ind1)>0)
        l = sum(1-Y(ind1).*(X(ind1,:)*w));
    end
    l2 = 0;
    if(length(ind2)>0)
        l2 = sum((1+h-Y(ind2).*(X(ind2,:)*w)).^2)./(4*h);
    end
    l = l+l2;
end

function g = gradHuberLoss(w,X,Y,h)
    [ind1,ind2] = huberInd(w,X,Y,h);
    g = zeros(size(w));
    if(length(ind1)>0)
        g = g - X(ind1,:)'*Y(ind1);
    end
    if(length(ind2)>0)
        g = g + (X(ind2,:)'*(X(ind2,:)*w) - (1+h)*X(ind2,:)'*Y(ind2))./(2*h);
    end
end

% end of file
