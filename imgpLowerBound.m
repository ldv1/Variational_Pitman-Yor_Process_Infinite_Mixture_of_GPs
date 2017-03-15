function lb = imgpLowerBound(model,vardist)

debug = 0;

N = model.N;
D = model.D;
C = vardist.C;

E_alpha = vardist.etahat1/vardist.etahat2;
E_log_alpha = digamma(vardist.etahat1)-log(vardist.etahat2);
E_log_vc = digamma(vardist.beta1)-digamma(vardist.beta1+vardist.beta2);
E_log_omvc = digamma(vardist.beta2)-digamma(vardist.beta1+vardist.beta2);
gammaSum = sum(vardist.gamma);
delta = vardist.delta;
eta1 = vardist.eta1;
eta2 = vardist.eta2;
sig2 = model.Likelihood.sigma2;
E_log_Rc = zeros(C,1);
log_det_W = zeros(C,1);
for c = 1:C
	log_det_W(c) = log(det(vardist.W(:,:,c)));
	E_log_Rc(c) = log_det_W(c) + D*log(2)+sum(digamma(0.5*(vardist.nu(c)+1-[1:D])));
end

%F1
F1 = -1/2*log(2*pi*sig2)*sum(gammaSum);
for c = 1:C
	F1 = F1 - 1/(2*sig2)* ...
	          vardist.gamma(:,c)'*( vardist.diagSigma(:,c) + (model.Y-vardist.mu(:,c)).^2 );
end

%F2
F2 = gammaSum(1:C-1)*E_log_vc;
for c = 2:C
	F2 = F2 + gammaSum(c)*sum(E_log_omvc(1:c-1));
end
F2 = F2 - D/2*log(2*pi)*sum(gammaSum) ...
        + 0.5*gammaSum*E_log_Rc;
for c = 1:C
	E_Rc = vardist.nu(c)*vardist.W(:,:,c);
	Xc = model.X' - repmat(vardist.g(:,c),1,N);
	F2 = F2 - 0.5*sum(Xc.*(E_Rc*Xc),1)*vardist.gamma(:,c) ...
	        - 0.5*trace(E_Rc*vardist.G(:,:,c))*gammaSum(c);
end

%F3
F3 = - (C-1)*gammaln(1-delta) ...
     + (C-1)*(1-delta)*E_log_alpha ...
	 - delta*sum(E_log_vc) ...
	 + (E_alpha+delta*[1:C-1]-1)*E_log_omvc;

%F4
F4 = eta1*log(eta2) - gammaln(eta1) + (eta1-1)*E_log_alpha - eta2*E_alpha;

%F6
F6 = -D*C/2*log(2*pi) + C/2*log(det(vardist.G0));
F6s = zeros(D,D);
for c = 1:C
	F6s = F6s + vardist.G(:,:,c) + (vardist.g(:,c)-vardist.g0)*(vardist.g(:,c)-vardist.g0)';
end
F6 = F6 - 0.5*trace(vardist.G0*F6s);

%F7
F7 = -sum(vardist.nu0)*D/2*log(2);
for c = 1:C
	F7 = F7 - 0.5*vardist.nu0(c)*log(det(vardist.W0(:,:,c))) ...
			- lmvgamma(D,vardist.nu0(c)/2) ...
			+ (vardist.nu0(c)-D-1)/2*E_log_Rc(c) ...
			- 0.5*trace(vardist.nu(c)*vardist.invW0(:,:,c)*vardist.W(:,:,c));
end

%E2
E2 = - sum(sum(vardist.gamma.*log(vardist.gamma+(vardist.gamma==0))));

%E3
E3 = - sum(gammaln(vardist.beta1+vardist.beta2)) ...
     + sum(gammaln(vardist.beta1)) + sum(gammaln(vardist.beta2)) ...
     - (vardist.beta1-1)'*E_log_vc - (vardist.beta2-1)'*E_log_omvc;

%E4
E4 = - vardist.etahat1*log(vardist.etahat2) ...
     + gammaln(vardist.etahat1) ...
	 - (vardist.etahat1-1)*E_log_alpha ...
		+ vardist.etahat2*E_alpha;

%E6
E6 = C*D/2*log(2*pi*exp(1));
for c = 1:C
	E6 = E6 + 0.5*log(det(vardist.G(:,:,c)));
end

%E7
E7 = D/2*(log(2)+1)*sum(vardist.nu);
for c = 1:C
	E7 = E7 + 0.5*vardist.nu(c)*log_det_W(c) ...
	        + lmvgamma(D,vardist.nu(c)/2) ...
		    - (vardist.nu(c)-D-1)/2*E_log_Rc(c);
end

%F5 and E5
F5plusE5 = C*N/2;
for c = 1:C
	Kc = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X);
	Bc = vardist.B(:,c);
	[ X L logdet ] = solve_chol_zeros(Kc, ...
	                                  Bc, ...
																		diag(Bc)+(model.Y-model.GP{c}.mean)*(vardist.mu(:,c)-model.GP{c}.mean)', ...
																		max(diag(Kc))+sig2+1e10);
	F5plusE5 = F5plusE5 - 0.5*logdet + 1/2*sum(log(Bc)) ...
	                    - 1/2*trace(X);
	
	% old implementation
	% Lc = chol(Kc+diag(Bc),'lower');
	% F5plusE5 = F5plusE5 - sum(log(diag(Lc))) + 1/2*sum(log(Bc)) ...
	%                    - 1/2*trace(Lc'\(Lc\(diag(Bc)+model.Y*vardist.mu(:,c)')));
	% checks
	% norm( X - Lc'\(Lc\(diag(Bc)+model.Y*vardist.mu(:,c)')) )
	% norm( 0.5*logdet - sum(log(diag(Lc))) )
end

LBparts = [ F1 F2 F3 F4 F6 F7 E2 E3 E4 E6 E7 F5plusE5 ];
lb = sum(LBparts);

assert( F4+E4 <= 1e-10 );
assert( F6+E6 <= 1e-10 );
assert( F7+E7 <= 1e-10 );
assert( F5plusE5 <= 1e-10 );

if any(isnan(LBparts))
	LBparts
	error('NaN detected in lower bound !');
end

if debug
	LBparts
end
