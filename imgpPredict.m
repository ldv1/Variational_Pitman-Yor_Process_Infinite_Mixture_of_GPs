function [ yp sig2 omega ypc ] = imgpPredict(model, vardist, Xtest, omega)

robotics = 0;

C = vardist.C; 
D = model.D;
N = model.N;
Nstar = size(Xtest, 1);

if nargin < 4
	omega = ones(C,1);
	for c = 2:C
		omega(c) = omega(c-1)*(1-vardist.beta1(c-1)/(vardist.beta1(c-1)+vardist.beta2(c-1)));
	end
	omega(1:C-1) = omega(1:C-1).*vardist.beta1(1:C-1)./(vardist.beta1(1:C-1)+vardist.beta2(1:C-1));
else
	disp('omega is _given_ for predictions');
	assert( length(omega) == C );
end
disp([ 'omega ' 'gamma ' 'sort(omega) ' 'sort(gamma) ' ]);
[ z1 t1 ] = sort(omega,'descend');
[ z2 t2 ] = sort(sum(vardist.gamma),'descend');
disp( [ omega sum(vardist.gamma)' t1 t2';  ] );
fprintf(1,'omega sum = %f\n', sum(omega));

yp = zeros(Nstar,1);
out1 = zeros(Nstar,1);

mix = gmm(D, C, 'full');
for c = 1:C
	mix.priors(c) = omega(c);
	mix.centres(c,:) = vardist.g(:,c)';
	mix.covars(:,:,c) = inv(vardist.W(:,:,c))/vardist.nu(c);
end
post = gmmpost(mix, Xtest);

for c = 1:C
	Kc = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X);
	[Kss, Kstar] = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X, Xtest);
	Bc = vardist.B(:,c);
	V = solve_chol_zeros(Kc, Bc, Kstar, max(diag(Kc))+model.Likelihood.sigma2+1e10);
	V = V';
	
	% old implementation
	% Lc = chol(Kc+diag(Bc),'lower');
	% V = (Lc'\(Lc\(Kstar)))';
	
	mustarc = model.GP{c}.mean + V*(model.Y(:)-model.GP{c}.mean);
	% wo/ approximation
	% mustarc = model.GP{c}.mean + V*(model.Y(:)-(Kc+diag(Bc))*inv(Kc+model.Likelihood.minSigma2*eye(N))*model.GP{c}.mean*ones(N,1));
	sigma2starc = Kss - sum(V.*Kstar',2) + model.Likelihood.sigma2;
	
	yp = yp + post(:,c).*mustarc;
	if robotics
		out1 = out1 + post(:,c).^2.*sigma2starc;
	else
		out1 = out1 + post(:,c).*(mustarc.^2+sigma2starc);
	end
end

if robotics
	sig2 = out1;
else
	sig2 = out1 - yp.^2;
end

if nargout > 2
	ypc = zeros(Nstar,C);
	
	for c = 1:C
		Kc = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X);
		[Kss, Kstar] = feval(model.GP{c}.covfunc, model.GP{c}.logtheta, model.X, Xtest);
		Bc = vardist.B(:,c);
		Lc = chol(Kc+diag(Bc),'lower');
		V = (Lc'\(Lc\(Kstar)))';
		ypc(:,c) = model.GP{c}.mean + V*(model.Y(:)-model.GP{c}.mean);
	end
end
