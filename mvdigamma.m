function y = mvdigamma(p,x)

y = zeros(size(x));
for d = 1:p
	y = y + digamma(x+(1-d)/2);
end
