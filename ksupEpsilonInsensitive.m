function [w,costs] = KsparseEpsReg(X,Y,lambda,k,eps,w0,h,iters_acc,eps_acc);
% Author: Matthew Blaschko - matthew.blaschko@inria.fr
% Copyright (c) 2013
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% performs epsilon insensitive regression with k-support regularization
% uses Huber smoothed epsilon insensitive regression
% first 3 arguments are required
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

if(~exist('h','var')) % Huber parameter (see e.g. Olivier
    % Chapelle. Training a Support Vector Machine in
    % the Primal, Neural Computation, 2007. Eq. (18))
    h = 0.1;
end

if(~exist('w0','var'))
    w0 = zeros(size(X,2),1);
end

if(~exist('k','var')) % have a sparsity factor of approx 1/4
    k = round(size(X,2)/4);
end

if(~exist('eps','var'))
    % to get standard Huber smothing of absolute loss, set eps to zero
    eps = 0;
end

if(size(X,1)>size(X,2)) % lipschitz constant for gradient of squared loss
    L = eigs(X'*X,1)/(2*h);
else
    L = eigs(X*X',1)/(2*h);
end
L = L+L; % could have contribution from both sides of loss if eps is zero

[w,costs] = overlap_nest(@(w)(epsInsensitiveLoss(w,X,Y,eps,h)),...
    @(w)(gradEpsInsensitiveLoss(w,X,Y,eps,h)), lambda, ...
    L, w0, k, iters_acc,eps_acc);
end

function [ind1,ind2] = huberEpsIndneg(w,X,Y,h,eps)
    ind1 = find(Y-X*w <= -eps-h);
    ind2 = find(abs(Y-X*w + eps) <=h);
end

function [ind1, ind2] = huberEpsIndpos(w,X,Y,h,eps)
    ind1 = find(Y - X*w >= eps+h);
    ind2 = find(abs(Y-X*w - eps)<=h);
end

function l = epsInsensitiveLoss(w,X,Y,eps,h)
    [ind1,ind2] = huberEpsIndneg(w,X,Y,h,eps);
    l = 0;
    if(length(ind1)>0)
        l = l + sum(-eps - Y(ind1) + X(ind1,:)*w);
    end
    if(length(ind2)>0)
        l = l + sum((Y(ind2) - X(ind2,:)*w + eps - h).^2)/(4*h);
    end
    [ind1,ind2] = huberEpsIndpos(w,X,Y,h,eps);
    if(length(ind1)>0)
        l = l + sum(Y(ind1) - X(ind1,:)*w - eps);
    end
    if(length(ind2)>0)
        l = l + sum((Y(ind2) - X(ind2,:)*w - eps + h).^2)/(4*h);
    end
end


function g = gradEpsInsensitiveLoss(w,X,Y,eps,h)
    [ind1,ind2] = huberEpsIndneg(w,X,Y,h,eps);
    g = zeros(size(w));
    if(length(ind1)>0)
        g = g + sum(X(ind1,:),1)'; % careful not to sum up if there is only one training sample
    end
    if(length(ind2)>0)
        g = g + X(ind2,:)'*(X(ind2,:)*w - eps + h - Y(ind2))/(2*h);
    end
    [ind1,ind2] = huberEpsIndpos(w,X,Y,h,eps);
    if(length(ind1)>0)
        g = g - sum(X(ind1,:),1)';
    end
    if(length(ind2)>0)
        g = g + X(ind2,:)'*(X(ind2,:)*w + eps - h - Y(ind2))/(2*h);
    end
end
