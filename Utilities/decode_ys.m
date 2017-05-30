function [decoded_Y] = decode_ys(Y)
    [n,d]=size(Y);
    yy = zeros(n,48);
    for i = 1:n
        A = Y(i,:)';
        B = reshape(A,3,48);
	yy(i,:) = sum(B);
    end
    [M, idx] = max(yy');
    decoded_Y = idx';
end
