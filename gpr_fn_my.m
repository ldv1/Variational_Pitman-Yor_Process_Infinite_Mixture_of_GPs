function [out1, out2] = gpr_fn_my(logtheta, covfunc, x, y, Bc, nonzero_m)

% gpr - Gaussian process regression, with a named covariance function. Two
% modes are possible: training and prediction: if no test data are given, the
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this mode is used to fit the hyperparameters.
% If test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = gpr(logtheta, covfunc, x, y)
%    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xstar)
%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%   mu       is a (column) vector (of size nn) of prediced means
%   S2       is a (column) vector (of size nn) of predicted variances
%
% For more help on covariance functions, see "help covFunctions".
%
% (C) copyright 2006 by Carl Edward Rasmussen (2006-03-20).
%

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed

[n, D] = size(x);
if nargin < 6 || ~nonzero_m
	nParams = length(logtheta);
	ac = zeros(n,1);
else
	nParams = length(logtheta)-1;
	ac = logtheta(end);
end

if eval(feval(covfunc{:})) ~= nParams
  error('Error: Number of parameters do not agree with covariance function')
end

Kc = feval(covfunc{:}, logtheta(1:nParams), x);
[ alpha Lc logdet ] = solve_chol_zeros(Kc, Bc, y-ac, max(diag(Kc))+1e10);

out1 = 0.5*sum((y-ac).*alpha) + 0.5*logdet + 0.5*n*log(2*pi);

if nargout == 2               % ... and if requested, its partial derivatives
  out2 = zeros(nParams,1);       % set the size of the derivative vector
  W = solve_chol_zeros(Kc, Bc, eye(n), max(diag(Kc))+1e10, Lc); % precompute for convenience
  W = W - alpha*alpha';
  for i = 1:nParams
    out2(i) = sum(sum(W.*feval(covfunc{:}, logtheta, x, i)))/2;
  end
	if nargin > 5 && nonzero_m % derivative w.r.t. the mean
		cc = solve_chol_zeros(Kc, Bc, ones(n,1), max(diag(Kc))+1e10, Lc);
		out2 = [ out2 ; cc'*(ac-y) ];
	end
  
end

