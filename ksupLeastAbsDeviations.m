function [w,costs] = KsparseLeastAbsDeviations(X,Y,lambda,k,w0,h,iters_acc,eps_acc);
% Author: Matthew Blaschko - matthew.blaschko@inria.fr
% Copyright (c) 2013
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% performs least absolute deviationss regression with k-support regularization
% uses Huber smoothing for differentiability of loss
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

% to get standard Huber smothing of absolute loss, set eps to zero
eps = 0;

[w,costs] = ksupEpsilonInsensitive(X,Y,lambda,k,eps,w0,h,iters_acc,eps_acc);

end
