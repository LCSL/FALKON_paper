function f = gaussianKernel_MWs(S1)
     f = @(X1,X2) exp(-.5*(SquareDist(X1,X2,S1)));
end

function D = SquareDist(X1, X2, S)
    R = chol(S);
    sq1 = sum((X1*R').^2,2);
    sq2 = sum((X2*R').^2,2);

    D = bsxfun(@plus,bsxfun(@plus, sq2', -2*X1*S*X2'), sq1);
end
