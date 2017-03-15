function [ model vardist lb ] = imgpTrain(X,Y,kernel,C,noise,delta,options)

debug = 0;
debug_grad = 0;
debug_init = 0;
bump = 0;

model.type = 'imgpmodel';

[ N D ] = size(X);
[ N Q ] = size(Y);
assert( size(X,1) == size(Y,1) );
model.X = X;
model.Y = Y;
model.N = N;	% no. of samples
model.Q = 1;	% output dimension
assert( model.Q == 1 );
model.D = D;	% input dimension
model.Likelihood.type = 'Gaussian';
model.Likelihood.noise = 'heterosc';	% all features have the same target noise
%model.Likelihood.sigma2 is initialized a bit later
model.Likelihood.minSigma2 = 1e-10;
if length(kernel) == 1
	kernel = repmat(kernel,1,C);
else
	if length(kernel) ~= C
		error('specify either a single kernel or as many kernel as C');
	end
end
model.GP = {};
for c = 1:C
	model.GP{c}.covfunc = kernel{c};
	switch model.GP{c}.covfunc
		case 'covSEard'
			dd = log((max(X)-min(X))'/2);	% log of length-scales, column vector
			dd(dd == -Inf) = 0;
			model.GP{c}.logtheta(1:D,1) = (1+0.1*(rand()-0.5))*(dd+0.1*(dd==0)) - mod(C,1)*rand();
			model.GP{c}.logtheta(D+1,1) = 0;	% the Y are supposed to be re-scaled as normally distributed,
												% so v0 = 1 is a good guess
		
		case 'covNNone'
			dd = log((max(X)-min(X))'/2);	% log of length-scales, column vector
			dd(dd == -Inf) = 0;
			model.GP{c}.logtheta(1,1) = min( dd - mod(C,1)*rand() );
			model.GP{c}.logtheta(2,1) = 0;	% the Y are supposed to be re-scaled as normally distributed,
											% so v0 = 1 is a good guess
			
		otherwise
			char(model.GP{c}.covfunc)
			error('Unknown covariance type')
	end
	model.GP{c}.nParams = length(model.GP{c}.logtheta);
	model.GP{c}.mean = 0.;
end
if bump
	model.GP{1}.logtheta(1:D,1) = (dd+0.1*(dd == 0));
	model.GP{2}.logtheta(1:D,1) = (dd+0.1*(dd == 0));
	model.GP{3}.logtheta(1:D,1) = (dd+0.1*(dd == 0)) - 4;
end

dispLB = options(1);	% display lower bound during training
dispEvery = 1;
kernLearn = options(2); % learn kernel hyperparameters (0 for not learning)
meanLearn = options(16); % learn the mean of the GPs
sigmaLearn = options(4);	% learn target noise (0 for not learning)
if sigmaLearn
	model.Likelihood.sigma2 = noise;
	if noise == 0
		model.Likelihood.sigma2 = mean(var(Y))*(5e-2)^2;	% 5pc
	end
	if debug
		fprintf(1,'sig2 will be learned: initial sig2 = %g\n', model.Likelihood.sigma2);
	end
else
	model.Likelihood.sigma2 = noise;
	if model.Likelihood.sigma2 < model.Likelihood.minSigma2
		fprintf(1,'resetting target noise to %g\n', model.Likelihood.minSigma2);
		model.Likelihood.sigma2 = model.Likelihood.minSigma2;
	end
	if debug
		fprintf(1,'sig2 is frozen to %g\n', model.Likelihood.sigma2);
	end
end
deltaLearn = options(6);	% learn delta (0 for not learning)
if deltaLearn
	vardist.delta = 0.1;
	if debug
		fprintf(1,'delta will be learned: initial delta = %g\n', vardist.delta);
	end
else
	vardist.delta = delta;
	if vardist.delta == 0
		fprintf(1,'resetting delta to 1e-8\n');
		vardist.delta = 1e-8;
	end
	if debug
		fprintf(1,'delta is frozen to %g\n', vardist.delta);
	end
end
nu0Learn = options(8);	% learn nu0 (0 for not learning)
W0Learn = options(9);	% learn W0 (0 for not learning)
labelReordering = options(10);
iter = options(11);	% number of variational EM iterations

learnKernEvery = 4;

% initialize factors

vardist.C = C;	% truncation threshold
assert( C <= N );
if options(15) == 0 % uniform initialization
	vardist.gamma = ones(N,C)/C;
	if meanLearn
		meanY = mean(Y);
		varY = var(Y);
		for c = 1:C
			model.GP{c}.mean = meanY + sqrt(varY)/10*randn();
		end
	end
else
	p = randperm(N);
	kmeans_options = foptions();
	if debug
		kmeans_options(1) = 1;
	end
	if options(15) == 1 % initialization using kmeans clustering in dimension D
		[mix.centres, kmeans_options, post] = kmeans(X(p(1:C),:), X, kmeans_options);
		vardist.gamma = post;	% hard assignments
	elseif options(15) == 2 % initialization using kmeans clustering in dimension D+1
		[mix.centres, kmeans_options, post] = kmeans([ X(p(1:C),:) Y(p(1:C)) ], [ X Y ], kmeans_options);
		vardist.gamma = post;	% hard assignments
	else % initialization using a GMM in dimension D+1
		mix = gmm(D+1,C,'full');
		kmeans_options(14) = 5; % just 5 iterations of K-means
		mix = gmminit(mix, [ X Y ], options);
		kmeans_options(14) = 15; % Max. number of iterations
		[mix, kmeans_options] = gmmem(mix, [ X Y ], kmeans_options);
		vardist.gamma = gmmpost(mix, [ X Y ]); % soft assignments
		[ m mid ] = max(vardist.gamma, [], 2);
	end
	
	if options(15) ~= 3
		post = logical(post);
	end
	
	if meanLearn
		for c = 1:C
			if options(15) ~= 3
				model.GP{c}.mean = mean(Y(post(:,c)));
			else
				model.GP{c}.mean = mix.centres(c,D+1);
			end
			fprintf(1,'mean of component %i: %f\n', c, model.GP{c}.mean);
		end
	end
	
	if debug_init && ( D == 1 )
		g = figure();
		hold on;
		plot( X, Y, '.', 'markersize', 12, 'color','black');
		for c = 1:C
			if options(15) ~= 3
				Xc = X(post(:,c),:);
				plot(Xc, c*ones(size(Xc)), 'markersize', 12, 'color','red');
			else
				ellipsoid = gellipse(mix.centres(c,:), mix.covars(:,:,c), 200);
				plot(ellipsoid(:,1), ellipsoid(:,2), '-r','linewidth', 5);
			end
		end
		% axis([min(X) max(X) min(min(Y),0) max(max(Y),C+1) ]);
		pause
	end
	
	vardist.gamma = vardist.gamma ./ repmat( sum(vardist.gamma,2), 1, C );
	vardist.gamma( vardist.gamma < 1e-40 ) = 1e-40;
end
if bump
	disp('cheating for the initialization of a bump signal');
	vardist.gamma = 1e-8*ones(N,C);
	vardist.gamma(1:N/4-1,1) = 1.;
	vardist.gamma(N/4+1:N,2) = 1.;
	vardist.gamma(N/4:N/4,3) = 1.;
	vardist.gamma = vardist.gamma ./ repmat( sum(vardist.gamma,2), 1, C );
end
vardist.mu = zeros(N,C);
vardist.diagSigma = zeros(N,C);
vardist.B = model.Likelihood.sigma2./vardist.gamma;
vardist.eta1 = 1e-3;	% vague Gamma prior if a << 1
vardist.eta2 = 1e-3;
vardist.beta1 = (1-vardist.delta+N/C)*ones(C-1,1);
vardist.beta2 = vardist.eta1/vardist.eta2*ones(C-1,1);
vardist.etahat1 = vardist.eta1;
vardist.etahat2 = vardist.eta2;
if debug
	fprintf(1,'E[alpha] = %f\n', vardist.etahat1/vardist.etahat2);
end
mu_x = mean(X); % row vector
R_x = zeros(D,D);
for n = 1:N
	R_x = R_x + (X(n,:)-mu_x)'*(X(n,:)-mu_x);
end
R_x = R_x / (N-1) / 10;
if debug
	fprintf(1,'estimated covariance matrix:\n');
	R_x
end
vardist.g0 = mu_x';
vardist.invG0 = R_x;
vardist.G0 = inv(R_x);
vardist.nu0 = D*ones(C,1);
vardist.W0 = repmat(inv(R_x)/D, [ 1 1 C ]);
if debug
	fprintf(1,'nu0 = %f\n', vardist.nu0(1));
	if D == 1
		fprintf(1,'W0 = %f\n', vardist.W0(1));
	end
end
for c = 1:C
	vardist.invW0(:,:,c) = inv(vardist.W0(:,:,c));
end
vardist.g = gsamp(vardist.g0, vardist.invG0, C)';	% D x C
vardist.G = repmat(vardist.invG0, [ 1 1 C]);
vardist.nu = vardist.nu0;
vardist.W = vardist.W0;	% D x D x C
if bump
	vardist.g(1) = -1;
	vardist.g(2) =  1;
	vardist.g(3) =  0.;
	vardist.G(1) =  0.03;
	vardist.G(2) =  0.03;
	vardist.G(3) =  0.03;
	vardist.W(1) =  1;
	vardist.W(2) =  1;
	vardist.W(3) =  1000;
	vardist.nu(1) =  N/2;
	vardist.nu(2) =  N/2;
	vardist.nu(3) =  5;
end
u = zeros(N,C);

% iterate

lb = zeros(iter,1);
LBold = -Inf;	% at the onset, the lower bound cannot be computed;
				% the reasong is that for the calculation of the lower bound
				% we assume that Sigmac = Kc inv(Kc+Bc) Bc and muc = Sigmac (inv(B) y + inv(Kc) ac)
				% but right at the onset muc = Sigmac = 0 

for niter = 1:iter
	
	gammaSum = sum(vardist.gamma);
	
	% E step
	
	vardist.B = model.Likelihood.sigma2./vardist.gamma;
	
	for c = 1:C
		Kc = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X);
		[ Cc L ] = solve_chol_zeros(Kc, vardist.B(:,c), diag(vardist.B(:,c)), max(diag(Kc))+model.Likelihood.sigma2+1e10);
		
		% old implementation
		% Cc = eye(N) + 1/model.Likelihood.sigma2*Kc.*(sqrt(vardist.gamma(:,c))*sqrt(vardist.gamma(:,c))');
		% LCc = chol(Cc,'lower');
		% T = 1/sqrt(model.Likelihood.sigma2)*(LCc\diag(sqrt(vardist.gamma(:,c))))*Kc;
		
		% update factor q(f_c)
		% we never store all the big matrices Sigma !
		
		% Sigma = Kc - T'*T;
		Sigma = Kc*Cc;
		% test against the old implementation
		% norm(Sigma-(Kc - T'*T))
		vardist.diagSigma(:,c) = diag(Sigma);
		vardist.mu(:,c) = Sigma*((1./vardist.B(:,c)).*Y);
		if meanLearn
			cc = solve_chol_zeros(Kc, vardist.B(:,c), model.GP{c}.mean*ones(N,1), max(diag(Kc))+model.Likelihood.sigma2+1e10, L);
			vardist.mu(:,c) = vardist.mu(:,c) + vardist.B(:,c).*cc;
		end
		
		% construct u for the update of factor q(z_nc)
		u(:,c) = -1/(2*model.Likelihood.sigma2)* ...
		         ( vardist.diagSigma(:,c) + (Y-vardist.mu(:,c)).^2 );
	end
	
	if debug
		fprintf(1,'E-step: lower bound after update of q(f) = %f\n', imgpLowerBound(model, vardist));
	end

	% update factor q(alpha)
	log_omvc = digamma(vardist.beta2) - digamma(vardist.beta1+vardist.beta2);
	vardist.etahat1 = vardist.eta1 + (C-1)*(1-vardist.delta);
	vardist.etahat2 = vardist.eta2 - sum(log_omvc);
	if debug
		fprintf(1,'E-step: lower bound after update of q(alpha): %f, E[alpha] = %g\n', imgpLowerBound(model, vardist), vardist.etahat1/vardist.etahat2);
	end
	
	% update factor q(v_c)
	vardist.beta1 = 1-vardist.delta+gammaSum(1:C-1)';
	vardist.beta2 = vardist.etahat1/vardist.etahat2 + vardist.delta*[1:C-1]';
	for c = 1:C-1
		vardist.beta2(c) = vardist.beta2(c) + sum(gammaSum(c+1:C));
	end
	if debug
		fprintf(1,'E-step: lower bound after update of q(v_c): %f\n', imgpLowerBound(model, vardist));
	end
	
	% update factor q(z_nc)
	% for c=C, E[log(v_c)] = 0 since v_C = 1
	for c = 1:C-1
		u(:,c) = u(:,c) + ...
		         digamma(vardist.beta1(c)) - digamma(vardist.beta1(c)+vardist.beta2(c));
	end
	for c = 2:C
		u(:,c) = u(:,c) + ...
		         sum( digamma(vardist.beta2(1:c-1)) - digamma(vardist.beta1(1:c-1)+vardist.beta2(1:c-1)) );
	end
	u = u - 1/2*log(2*pi*model.Likelihood.sigma2) - D/2*log(2*pi);
	for c = 1:C
		E_Rc = vardist.nu(c)*vardist.W(:,:,c);
		E_log_Rc = log(det(vardist.W(:,:,c))) + D*log(2)+sum(digamma(0.5*(vardist.nu(c)+1-[1:D])));
		Xc = X' - repmat(vardist.g(:,c),1,N);
		u(:,c) = u(:,c) + ...
		         0.5*E_log_Rc - 0.5*( trace(vardist.G(:,:,c)*E_Rc) + sum(Xc.*(E_Rc*Xc),1)' );
	end
	for n = 1:N
		[ mx idmx ] = max(u(n,:));
		vardist.gamma(n,:) = exp(u(n,:)-mx)./sum(exp(u(n,:)-mx));
	end
	vardist.gamma( vardist.gamma<1e-30 ) = 1e-30;
	if debug
		fprintf(1,'E-step: lower bound after update of q(z_nc): %f\n', imgpLowerBound(model, vardist));
	end
	
	% label re-ordering
	
	if labelReordering
		[ dummy I ] = sort(sum(vardist.gamma),'descend');
		
		vardist.gamma = vardist.gamma(:,I);
		vardist.diagSigma = vardist.diagSigma(:,I);
		vardist.mu = vardist.mu(:,I);
		vardist.G = vardist.G(:,:,I);
		vardist.g = vardist.g(:,I);
		vardist.W = vardist.W(:,:,I);
		vardist.nu = vardist.nu(I);
		
		vardist.B = model.Likelihood.sigma2./vardist.gamma;
		
		if debug
			fprintf(1,'E-step: lower bound after re-ordering of q(z_nc): %f\n', imgpLowerBound(model, vardist));
		end
	end
	
	% update factor q(m_c)
	for c = 1:C
		E_Rc = vardist.nu(c)*vardist.W(:,:,c);
		vardist.G(:,:,c) = inv( vardist.G0 + sum(vardist.gamma(:,c))*E_Rc);
		vardist.g(:,c) = vardist.G(:,:,c)*(vardist.G0*vardist.g0 + E_Rc*sum(repmat(vardist.gamma(:,c)',D,1).*X',2));
	end
	if debug
		fprintf(1,'E-step: lower bound after update of q(m_c): %f\n', imgpLowerBound(model, vardist));
	end
	
	% update factor q(R_c)
	for c = 1:C
		vardist.nu(c) = vardist.nu0(c) + sum(vardist.gamma(:,c));
		vardist.W(:,:,c) = vardist.invW0(:,:,c);
		for n = 1:N
			vardist.W(:,:,c) = vardist.W(:,:,c) + vardist.gamma(n,c)*(X(n,:)'-vardist.g(:,c))*(X(n,:)'-vardist.g(:,c))';
		end
		vardist.W(:,:,c) = inv( vardist.W(:,:,c) + sum(vardist.gamma(:,c))*vardist.G(:,:,c) );
	end
	if debug
		fprintf(1,'E-step: lower bound after update of q(R_c): %f\n', imgpLowerBound(model, vardist));
	end
	
	
	% M step
	
	if sigmaLearn
		s = 0.;
		for c = 1:C
			s = s + (vardist.diagSigma(:,c) + (Y-vardist.mu(:,c)).^2)'*vardist.gamma(:,c);
		end
		model.Likelihood.sigma2 = s/sum(vardist.gamma(:));
	end
	if debug
		fprintf(1,'M-step: lower bound after optimization of sig2: %f, sig2 = %f\n', imgpLowerBound(model, vardist), model.Likelihood.sigma2);
	end
	
	if deltaLearn
		if debug_grad
			checkgrad('imgp_delta', log(vardist.delta/(1-vardist.delta)), 1e-4, vardist);
		end
		[ tfdelta fX ] = minimize(log(vardist.delta/(1-vardist.delta)), 'imgp_delta', 5, vardist);
		vardist.delta = exp(tfdelta)/(1+exp(tfdelta));
		if debug
			fprintf(1,'M-step: lower bound after optimization of delta: %f, delta = %f\n', imgpLowerBound(model, vardist), vardist.delta);
		end
	end
	
	if debug_grad
		checkgrad('imgp_eta1', log(vardist.eta1), 1e-4, vardist);
	end
	[ logeta1  fX ] = minimize(log(vardist.eta1) , 'imgp_eta1' , 5, vardist);
	vardist.eta1 = exp(logeta1);
	if debug
		fprintf(1,'M-step: lower bound after optimization of eta1: %f, eta1 = %f\n', imgpLowerBound(model, vardist), vardist.eta1);
	end
	
	vardist.eta2 = vardist.eta1*vardist.etahat2/vardist.etahat1;
	if debug
		fprintf(1,'M-step: lower bound after optimization of eta2: %f, eta2 = %f\n', imgpLowerBound(model, vardist), vardist.eta2);
	end
	
	if nu0Learn
		if debug_grad
			checkgrad('imgp_nu0', log(vardist.nu0-(D-1)), 1e-4, vardist, model);
		end
		[ tfnu0  fX ] = minimize(log(vardist.nu0-(D-1)) , 'imgp_nu0' , 5, vardist, model);
		vardist.nu0 = exp(tfnu0)+(D-1);
		if debug
			fprintf(1,'M-step: lower bound after optimization of nu0: %f, nu0(1) = %f\n', imgpLowerBound(model, vardist), vardist.nu0(1));
		end
	end
	
	if ( D == 1 ) && W0Learn
		S = zeros(D,D);
		for c = 1:C
			vardist.W0(:,:,c) = vardist.nu(c)/vardist.nu0(c)*vardist.W(:,:,c);
			vardist.invW0(:,:,c) = inv(vardist.W0(:,:,c));
		end
		if debug
			fprintf(1,'M-step: lower bound after optimization of W0: %f, W0(:,:,c) = %f\n', imgpLowerBound(model, vardist), vardist.W0(:,:,c));
		end
	end
	
	if kernLearn && ( mod(niter,learnKernEvery) == 0 )
		
		fprintf(1,'optimizing the hyperparameters\n');
		
		vardist.B = model.Likelihood.sigma2./vardist.gamma;
		
		for c = 1:C
			if model.GP{c}.nParams > 0
				
				% kernel hyperparameters
				if meanLearn
					logtheta = [ model.GP{c}.logtheta(:) ; model.GP{c}.mean ];
					if debug_grad
						checkgrad('gpr_fn_my', logtheta, 1e-4, model.GP{c}.covfunc, model.X, model.Y, vardist.B(:,c), 1);
					end
					[logtheta fX] = minimize(logtheta, 'gpr_fn_my', 5, model.GP{c}.covfunc, model.X, model.Y, vardist.B(:,c), 1);
					model.GP{c}.logtheta = logtheta(1:end-1);
					model.GP{c}.mean = logtheta(end);
				else
					if debug_grad
						checkgrad('gpr_fn_my', model.GP{c}.logtheta(:), 1e-4, model.GP{c}.covfunc, model.X, model.Y, vardist.B(:,c));
					end
					[logtheta fX] = minimize(model.GP{c}.logtheta(:), 'gpr_fn_my', 5, model.GP{c}.covfunc, model.X, model.Y, vardist.B(:,c));
					model.GP{c}.logtheta = logtheta;
				end
				
				% update q(f_c)
				Kc = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X);
				Cc = solve_chol_zeros(Kc, vardist.B(:,c), diag(vardist.B(:,c)), max(diag(Kc))+model.Likelihood.sigma2+1e10);
				Sigma = Kc*Cc;
				vardist.diagSigma(:,c) = diag(Sigma);
				vardist.mu(:,c) = Sigma*((1./vardist.B(:,c)).*Y);
				if meanLearn
					cc = solve_chol_zeros(Kc, vardist.B(:,c), model.GP{c}.mean*ones(N,1), max(diag(Kc))+model.Likelihood.sigma2+1e10); % inv(Kc+Bc)*ac*I
					vardist.mu(:,c) = vardist.mu(:,c) + vardist.B(:,c).*cc;
				end
				if debug
					fprintf(1,'M-step: lower bound after update of K_%i: %f\n', c, imgpLowerBound(model, vardist));
				end
			end
		end
		
		if debug
			theta = [];
			for c = 1:C
				theta = [ theta exp(model.GP{c}.logtheta(:)) ];
			end
			theta
		end
	end
	
	% print lower bound
	
	if dispLB == 1
		LBnew = imgpLowerBound(model, vardist);
		fprintf(1,'Iteration%4d/%4d  Fnew %11.6f Fold %11.6f Diffs %20.12f\n', ...
		        niter, iter, LBnew, LBold,  LBnew-LBold);
		if ( LBnew < LBold ) && ~labelReordering
			error('non-increasing lower bound !');
		end
		if ( LBnew >= LBold ) && ( LBnew < LBold + 1e-4 ) && ( niter > 1 )
			fprintf(1,'relative increase of lower bound below 1e-4\n');
			return;
		end
		LBold = LBnew;
		lb(niter) = LBnew;
	else
		if mod(niter,dispEvery) == 0
			fprintf(1,'Iteration%4d/%4d\n',niter,iter);
		end
	end
end
