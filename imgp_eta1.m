function [f, df] = imgp_eta1(logeta1, vardist)

eta1 = exp(logeta1);
C = vardist.C;
E_log_alpha = digamma(vardist.etahat1)-log(vardist.etahat2);

f = eta1*log(vardist.eta2) ...
    -gammaln(eta1) ...
		+eta1*E_log_alpha;
f = -f;

if nargout > 1
	df = log(vardist.eta2)-digamma(eta1)+E_log_alpha;
	df = df*eta1;
	df = -df;
end
