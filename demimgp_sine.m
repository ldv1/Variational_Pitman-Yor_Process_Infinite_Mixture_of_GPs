addpath misc_toolbox/;
addpath misc_toolbox/gpml/;
addpath misc_toolbox/netlab/;

randn('state', 1724);
rand('state',  1724);

n = 150;
s2 = 0.01;
alpha = 1;
beta = 2.5;
X = linspace(0,2,n)';
Y = sin(alpha*pi*X.^beta) + sqrt(s2)*randn(n,1);
n_test = 500;
Xtest = linspace(0,2,n_test)';
Ytest = sin(alpha*pi*Xtest.^beta);

options = [];
options(1) = 1; % display lower bound
options(2) = 1; % learn kernel hyperparameters
options(4) = 1; % learn target noise
options(6) = 1; % learn delta
options(8) = 1; % learn nu0
options(9) = 1; % learn W0
options(10) = 1; % label re-ordering
options(11) = 30; % no. of iterations
options(15) = 1;  % use Kmeans for the initialization
options(16) = 1;  % non-zero mean GPs
C = 20;	% threshold

% im-gp

ctime = cputime;
[ model vardist lb ] = imgpTrain(X,Y,{'covSEard'},C,s2,0.,options);
fprintf(1, 'training of the IM-GP completed in %f s.\n', cputime-ctime);
fprintf(1,'im-gp delta = %g\n', vardist.delta);
fprintf(1,'im-gp noise = %g\n', model.Likelihood.sigma2);
[ yp sig2 omega ypc ] = imgpPredict(model, vardist, Xtest);
sig = sqrt(sig2);

%disp('Assignments:');
%vardist.gamma

disp('nu0:');
for c = 1:C
	fprintf(1,'%f %f\n', vardist.nu0(c), vardist.W0(:,:,c));
end

% vanilla gp

ycovfunc = {'covSum', {'covSEard', 'covNoise'}};
logtheta = [ log(0.5), log(1), 0.5*log(s2) ];
% checkgrad('gpr_fn', logtheta(:), 1e-4, ycovfunc, X, Y);
[logtheta fX] = minimize(logtheta(:), 'gpr', 5, ycovfunc, X, Y);
sig2vanilla = exp(2*logtheta(3));
fprintf(1,'vanilla-gp noise = %g\n', sig2vanilla);
logtheta = logtheta(1:2);
exp(logtheta)
K = feval('covSEard', logtheta, model.X);
[Kss, Kstar] = feval('covSEard', logtheta, model.X, Xtest);
Lc = chol(K+s2*eye(n),'lower');
V = (Lc'\(Lc\(Kstar)))';
yvanilla = V*model.Y;
sig2vanilla = Kss - sum(V.*Kstar',2) + sig2vanilla;
sigvanilla = sqrt(sig2vanilla);


fh1 = figure(1,"position",[0,0,900,1000]); % w,h

dots_size = 10; % 14
mean_size = 3; % 10
sig_size = 2;	% 3

subplot(4,1,1);

hold on
plot(X, Y,  '.', 'markersize', dots_size, 'color','black');
xlabel('Input')
ylabel('Target')
plot(Xtest,Ytest, 'color', 'black','linewidth', mean_size);
plot(Xtest, yp, '-b','linewidth', mean_size);
plot(Xtest, yp+(2*sig), '-b','linewidth', sig_size);
plot(Xtest, yp-(2*sig), '-b','linewidth', sig_size);
legend('data', 'function', 'PYP-GP', 'PYP-GP lower bound', 'PYP-GP upper bound');
hold off
axis([0 2 -1.5 2.5]);
t = ['N=' num2str(n) ', C=' num2str(C) ', target noise=' num2str(s2) ];
if options(4)
	t = [ t ' (learned)' ];
else
	t = [ t ' (frozen, i.e. not learned)' ];
end
%title(t);


subplot(4,1,2);

hold on
plot(X, Y,  '.', 'markersize', dots_size, 'color','black');
xlabel('Input')
ylabel('Target')
plot(Xtest, Ytest, 'color', 'black','linewidth', mean_size);
plot(Xtest, yvanilla, '-r','linewidth', mean_size);
plot(Xtest, yvanilla+(2*sigvanilla), '-r','linewidth', sig_size);
plot(Xtest, yvanilla-(2*sigvanilla), '-r','linewidth', sig_size);
legend('data', 'function', 'Vanilla-GP', 'Vanilla-GP lower bound', 'Vanilla-GP upper bound');
hold off
axis([0 2 -1.5 2.5]);


subplot(4,1,3);

hold all
plot(X, Y,  '.', 'markersize', dots_size, 'color','black');
xlabel('Input')
ylabel('Target')
plot(Xtest, Ytest, 'color', 'black','linewidth', mean_size);
t = { 'samples' 'true curve' };
k = 3;
col = { 'r', 'g', 'b', 'r', 'b' };
for c = 1:C
	if omega(c) > 0.1
		plot(Xtest, ypc(:,c), 'linewidth', mean_size, 'color', col{mod(c,4)+1});
		t{k} = num2str(omega(c));
		k = k + 1;
	end
end
for c = 1:C
	if omega(c) > 0.1
		fprintf(1,'plotting component c = %g: 2*sig = %f, W = %g, nu = %f, G = %f\n', ...
		          c, 2*sqrt(inv(vardist.W(:,:,c))/vardist.nu(c)), vardist.W(:,:,c), vardist.nu(c), vardist.G(:,:,c));
	end
end
legend(t);
axis([0 2 -1.5 4]);
hold off

subplot(4,1,4);

hold all
k = 3;
for c = 1:C
	if omega(c) > 0.1
		plot(Xtest, gauss(vardist.g(:,c), inv(vardist.W(:,:,c))/vardist.nu(c), Xtest), 'linewidth', mean_size, 'color', col{mod(c,4)+1});
		t{k} = num2str(omega(c));
		k = k + 1;
	end
end
axis([0 2 -1.5 4]);
hold off

fprintf(1,'rms on im-gp: %f\n', norm(Ytest-yp)/sqrt(n_test));
fprintf(1,'rms on vanilla gp: %f\n', norm(Ytest-yvanilla)/sqrt(n_test));

% print -dpng  a.png -S2304,640

disp(' ')
disp('Press any key to end.')
pause
close(fh1);
clear all;
