alpha = 4;
C = 10;
maxs2 = 1.;
v = betarnd(1,alpha,C,1);
v(C) = 1.;
color_mean = unifrnd(0,255,C,3);
color_s2 = zeros(3,3,C);
for c = 1:C
	color_s2(:,:,c) = diag(unifrnd(0,maxs2,3,1));
end
omega = ones(C,1);
for c = 2:C
	omega(c) = omega(c-1)*(1-v(c-1));
end
omega(1:C-1) = omega(1:C-1).*v(1:C-1);

n = 64;
im = zeros(n,n,3);

for p = 1:n
	for q = 1:n
		u = rand();
		for c = 1:C
			if u <= omega(c)
				break;
			end
		end
		
		im(p,q,:) = gsamp(color_mean(c,:), color_s2(:,:,c), 1);
	end
end

h = figure();
imagesc(im);

pause;
close(h);
clear all;

