function D = gellipse(mu, sigma, n)

mu = mu(:);

T = linspace(0,2*pi,n)';
X = cos(T);
Y = sin(T);

[ U L ] = eig(sigma);
E = repmat(mu,1,n) + U*2*sqrt(L)*[ X Y ]';
D = E';
