function p = KtsProd(X, C, alpha, blk, kern)
    n = size(X,1); m = size(C,1);
    ms = ceil(linspace(0, n, blk+1));
    p = zeros(n, size(alpha,2) , 'like', C);
    if issparse(X) && strcmp(kern.func, 'linear')
        coeff2 = 1/kern.param2^2;
        coeff1 = kern.param1;
        %coeff1 = 1;
        %coeff2 = 1/kern.param^2;
        p = coeff1*ones(size(X,1),1)*sum(alpha,1)+ coeff2*X*(C'*alpha);
    else
%         for i=1:blk 
%             clear Kr;
%             Kr = (feval(class(C), X((ms(i)+1):ms(i+1), :)), C);
%             p((ms(i)+1):ms(i+1), :) = Kr*alpha;
% 
%             fprintf('*');
%         end
%         p = ones(size(X,1),size(C,1))*alpha + X*(C'*alpha);
%     else
        for i=1:blk 
            clear Kr;
            Kr = kern(feval(class(C), X((ms(i)+1):ms(i+1), :)), C);
            p((ms(i)+1):ms(i+1), :) = Kr*alpha;

            fprintf('*');
        end
    end
    fprintf('\n');
end
