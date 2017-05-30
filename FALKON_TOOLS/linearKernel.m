function f = linearKernel(beta, sigma)
    f = @(X1, X2) kk(X1, X2, beta, sigma);
end

function K = kk(X1, X2, beta, sigma)
	s = 1/(sigma^2);
	K = X1*X2';
	K = K*s;
	K = K + beta;
end
