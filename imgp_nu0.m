function [f, df] = imgp_nu0(tfnu0, vardist, model)

D = model.D;
C = vardist.C;
E_log_Rc = zeros(C,1);
log_det_W = zeros(C,1);
log_det_W0 = zeros(C,1);
for c = 1:C
	log_det_W0(c) = log(det(vardist.W0(:,:,c)));
	log_det_W(c) = log(det(vardist.W(:,:,c)));
	E_log_Rc(c) = log_det_W(c) + D*log(2)+sum(digamma(0.5*(vardist.nu(c)+1-[1:D])));
end

nu0 = exp(tfnu0)+(D-1);	% in (D-1: \infty);

f = -sum(nu0)*D/2*log(2) ...
    -nu0'/2*log_det_W0 ...
	-sum(lmvgamma(D,nu0/2)) ...
	+(nu0'-D-1)/2*E_log_Rc;
f = -f;

if nargout > 1
	df = -D/2*log(2) ...
    	 -1/2*log_det_W0 ...
		 -1/2*mvdigamma(D,nu0/2) ...
		 +1/2*E_log_Rc;
	df = df.*(nu0-(D-1));
	df = -df;
end
