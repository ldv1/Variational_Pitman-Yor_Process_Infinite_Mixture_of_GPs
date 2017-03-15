function [ X L logdet r ] = solve_chol_zeros(M, D, Y, e, L)

verbose = 0;
no_approx = 0;

% we solve (M + D) x = Y
% where D is diagonal (thus D is given as a vector)
% the elements of D that are bigger than a given threshold e are set to \infty

if nargin ~= 0
	
	ctime = cputime;
	
	[ n n ] = size(M);
	assert( length(D) == n );
	assert( size(Y,1) == n );
	
	if no_approx
		
		r = n;
		A = M + diag(D);
		L = chol(A,'lower');
		X = L'\(L\Y);
		logdet = 2*sum(log(diag(L)));
		
	else
	
		p = (D < e);
		r = sum(p);
		s = n-r;
		assert( s == sum(~p) );
		invD = 1./D(~p);	% of size s
		A = M(p,p) + diag(D(p));
		B = M(p,~p);	% of size r x s
		if nargin < 5
			L = chol(A,'lower');
		end
		Y1 = Y(p,:);
		Y2 = Y(~p,:);
		U = L'\(L\Y1);
		diag_invD_B_t = bsxfun(@times,B',invD); % = diag(invD)*B';
		V1 = -L'\(L\(diag_invD_B_t'*Y2));
		V2 = -diag_invD_B_t*U;	% = -diag(invD)*B'*U
		W = bsxfun(@times,Y2,invD); % = diag(invD)*Y2;
		X = zeros(size(Y));
		if size(X,2) ~= 1
			X(p,:) = U+V1;
			X(~p,:) = V2 + W;
		else
			X(p) = U+V1;
			X(~p) = V2+W;
		end

		logdet = sum(log(D(~p))) + 2*sum(log(diag(L)));
		
	end
		
	if verbose
		fprintf(1, 'solve_chol_zeros: n = %d instead of %d (threshold = %g) completed in %f s.\n', r, n, e, cputime-ctime);
		toc;
	end

else	% do a test
	
	n = 15000;
	m = 2000;	% m diagonal elements are set to a very large number
	A = rand(n,n);
	A = A*A';
	D = rand(n,1);
	p = randperm(n);
	e = 1e14;
	D(p(1:m)) = 1e15;
	y = rand(n,2);
	
	ctime = cputime;
	tic
	[ x1 L logdet q ] = solve_chol_zeros(A, D, y, e);
	fprintf(1, 'solve_chol_zeros completed in %f s.\n', cputime-ctime);
	toc
	tic
	B = A + diag(D);
	
	ctime = cputime;
	L = chol(B,'lower');
	x2 = L'\(L\y);
	logdet2 = 2*sum(log(diag(L)));
	fprintf(1, 'usual method completed in %f s.\n', cputime-ctime);
	toc
	
	fprintf(1,'test: error on the solution of Ax=b: %g\n', norm(x1 - x2));
	fprintf(1,'test: log(det(A)) = %g, error on the log(det(A)): %g\n', logdet2, logdet2-logdet);
	
end
