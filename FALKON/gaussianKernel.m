function f = gaussianKernel(sigma)
    f = @(X1,X2) res(X1, X2, sigma);
end


function D = res(X1, X2, sigma)

    sq1 = sum(X1.^2,2);

    sq2 = sum(X2.^2,2)';

    D = X1*X2';

    clear X2
    clear X1

    D = -2.0*D;

    D = bsxfun(@plus, D, sq2);
    clear sq2

    D = bsxfun(@plus, D, sq1);
    clear sq1

	D = bsxfun(@times, D, -1/(2*sigma^2));

    if isa(D, 'gpuArray')
        D = arrayfun(@exp, D);
    else
        D = exp(D);
    end
end
