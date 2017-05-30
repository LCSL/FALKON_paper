function [encoded_Y] = encode_ys(Y)
    n_classes = 144;
    a = (n_classes-2)/n_classes;
    a = 1;
    b = -1/(n_classes-1);
    ys = Y + 1;
    n = size(Y,1);
    encoded_Y = ones(n, n_classes)*b;
    for i = 1:n
        encoded_Y(i,ys(i)) = a;
    end
end
