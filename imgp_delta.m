function [f, df] = imgp_delta(tfdelta, vardist)

delta = exp(tfdelta)/(1+exp(tfdelta));	% in (0,1)
C = vardist.C;
E_log_alpha = digamma(vardist.etahat1)-log(vardist.etahat2);
E_log_vc = digamma(vardist.beta1)-digamma(vardist.beta1+vardist.beta2);
E_log_omvc = digamma(vardist.beta2)-digamma(vardist.beta1+vardist.beta2);

f = -(C-1)*gammaln(1-delta) ...
    +(C-1)*(1-delta)*E_log_alpha ...
		+delta*(-sum(E_log_vc) + [1:C-1]*E_log_omvc);
f = -f;

if nargout > 1
	df = (C-1)*digamma(1-delta) ...
    	 -(C-1)*E_log_alpha ...
			 +(-sum(E_log_vc) + [1:C-1]*E_log_omvc);
	df = df*(delta*(1-delta));
	df = -df;
end
