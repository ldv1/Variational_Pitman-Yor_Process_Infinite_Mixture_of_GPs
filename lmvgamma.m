function y = lmvgamma(p,x)

% p is a small strictly positive integer, this is the dimension of the input space
% the dimension of n is C, thus may be large

y = zeros(size(x));
y = p*(p-1)/4*log(pi);
for d = 1:p
	y = y + gammaln(x+(1-d)/2);
end

