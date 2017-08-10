function [alpha] = falkon(X, C, kernel, y, lambda, T, cobj, callback, memToUse, useGPU)
    n = size(X,1); m = size(C,1);

	isXsparse = issparse(X);
    if isXsparse
		kernMemCoeff = 2.0;
    elseif useGPU
    	kernMemCoeff = 2.0;
    else
        kernMemCoeff = 1.1;
    end

    if issparse(X) && ~(isstruct(kernel) && strcmp(kernel.func, 'linear'))
        ggg = gcp();
        ggg.IdleTimeout = Inf;
        clear ggg;
        X = distributed(X);
        spmd
            X = redistribute(X, codistributor1d(1));
        end
    end


    if ~exist('useGPU','var') || isempty(useGPU)
        useGPU = 0;
        disp('CPU will be used.');
    end

    if ~exist('memToUse', 'var') || isempty(memToUse)
        meminfo = memoryInfo();
        freeMem = double(meminfo.free)*1024;
        memToUse = 0.985*freeMem;
        disp(['No memory limit specified. At most ', num2str(memToUse/10^9), 'GB of RAM will be used']);
    else
        memToUse = memToUse * 10^9;
    end

    disp(''); disp('Precondition computation...');

    tic;
    [factKnmP, kern, freeDoubles] = computeMemory(memToUse, kernel, kernMemCoeff, n, m, size(y,2), size(C,2), isXsparse, useGPU);
    [cholR, cholRt, cholA, cholAt] = createRA(kern, C, m, lambda, freeDoubles, useGPU);
    toc

    disp(''); disp('Iterations...');
    tic;
    KnmP = factKnmP(X, C);
    b = cholAt(cholRt(KnmP([], double(y)/double(n))));
    mW = @(z) multW(KnmP, lambda, z, n, cholR, cholRt, cholA, cholAt);
    ncb = @(c, cobj) callback(cholR(cholA(c)), cobj);
    c = conjgrad(mW, b, T, [], cobj, ncb);
    time_cg = toc;
    disp(['Iterations required ', num2str(time_cg), ' seconds']);

    alpha = cholR(cholA(c));
end


function [fKnmP, kern, freeDoubles] = computeMemory(memToUse, kernel, kernMemCoeff, n, m, dim, dim_c, isXsparse, useGPU)

    freeDoubles = memToUse/8.0 - m*m - 4*m*dim - 2*n*dim;

    if freeDoubles < 0.1*m*m
        error(['The RAM memory you required to use, is to small w.r.t. the Nystrom centers.', ...
            'For the selected number of centers at least ', num2str((1.1*m*m + 4*m*dim + 2*n*dim)*8.0/1e9), 'GB of RAM are needed.']);
    end


    if isstruct(kernel) && strcmp(kernel.func, 'linear')
        kern = @(X1, X2) blockKernComp(X1, X2, linearKernel(kernel.param1, kernel.param2), kernMemCoeff, freeDoubles, useGPU);
    else
        kern = @(X1, X2) blockKernComp(X1, X2, kernel, kernMemCoeff, freeDoubles, useGPU);
    end

    DMIN = 64;

    if useGPU && ~isXsparse
        gD = gpuDevice();
        dbmem = gD.AvailableMemory/8;
        d = dim_c;
        dim_gpu = floor((dbmem - m*(d+4*dim) - 2*n*dim)/(d + kernMemCoeff*m));

        dim_cpu = freeDoubles/m;

        if dim_gpu >= DMIN
            blk = ceil(n/min(dim_cpu, dim_gpu));
        else
            blk = ceil(n/dim_cpu);
        end
    else
        blk = ceil(n*m*kernMemCoeff/freeDoubles);
    end

    function KnmP = factorySimpleKnmP(X, C, kernel, kernMemCoeff, ff, useGPU)
        Knm = blockKernComp(X, C, kernel, kernMemCoeff, ff, useGPU);
        KnmP = @(u,v) KtKprod(Knm, u, v);
    end

    function KnmP = factoryCpuKnmP(X, C)
        KnmP = @(u,v) KnmProd(X, C, u, v, blk, kern);
    end

    function p = linearKnmP(X, C, beta, sigma, u, z)
            bt = (numel(u)>0) + 2*(numel(z) > 0);
            coeff = 1/sigma^2;
            if bt == 0
                p = 0;
                return;
            end

            switch bt
                case 1
                    u = beta*ones(size(X,1),1)*sum(u,1) + X*(C'*u)*coeff;
                    p = (beta*sum(u,1)'*ones(1,size(C,1)) + ((u'*X)*C')*coeff)';
                case 2
                    p = (beta*sum(z,1)'*ones(1,size(C,1)) + ((z'*X)*C')*coeff)';
                case 3
                    z = z + beta*ones(size(X,1),1)*sum(u,1) + (X*(C'*u))*coeff;
                    p = (beta*sum(z,1)'*ones(1,size(C,1)) + ((z'*X)*C')*coeff)';
            end
            disp('*');
    end


    if isstruct(kernel) && strcmp(kernel.func, 'linear')
        fKnmP = @(X, C) @(u, v) linearKnmP(X, C, kernel.param1, kernel.param2, u, v);
    elseif 0.985*freeDoubles >= n*m
        fKnmP = @(X, C) factorySimpleKnmP(X, C, kernel, kernMemCoeff, freeDoubles - n*m, useGPU);
    elseif useGPU && dim_gpu >= DMIN
        fKnmP = @(X, C) @(u,v) gather(KnmProd(X, gpuArray(C), gpuArray(u), gpuArray(v), blk, kernel));
    else
        fKnmP = @factoryCpuKnmP;
    end
end

function [cholR, cholRt, cholA, cholAt] = createRA(kern, C, m, lambda, freeDoubles, useGPU)
    %this function computes R, A
    %R = chol(Z + 1e-15*m*eye(m));
    %A = chol(cast(R*R'/m + lambda*eye(m)));

    tic;
        Z = kern(C, []);
    toc;

    inplace_chol(Z, 0, 1.0, 1e-15*sum(diag(Z)));
    dR = diag(Z);


    %next lines do tril(Z) = triu(Z)*triu(Z)'
    if useGPU
        gD = gpuDevice();
        dbmem = gD.AvailableMemory/8;
        nmax = min(dbmem/(3.0*m), sqrt(freeDoubles/3.0));
    else
        nmax = freeDoubles/(3.0*m);
    end

    [download, upload] = produceDU(useGPU, 0);

    blk = ceil(m/nmax);
    ms = ceil(linspace(0, m, blk + 1));

    for i = 1:blk
        for j = blk:-1:i
            inti = (ms(i)+1):ms(i+1); intj = (ms(j)+1):ms(j+1);
            if i == j
                C1 = upload(triu(Z(intj, intj)));
                C2 = upload(triu(Z(intj, intj)));
                A = download(C1*C2');
                clear C1
                clear C2
            else
                C1 = upload(Z(inti, intj));
                C2 = upload(triu(Z(intj, intj)));
                A = download(C1*C2');
                clear C1
                clear C2
            end
            if j < blk
                longj = (ms(j+1)+1):m;
                C1 = upload(Z(inti, longj));
                C2 = upload(Z(intj, longj));
                A = A + download(C1*C2');
                clear C1
                clear C2
            end
            if i == j
                Z(intj, inti) = triu(A)' + triu(Z(intj, inti),1);
            else
                Z(intj, inti) = A';
            end
        end
    end

    %next lines do Z(:,:) = Z';
    for i = 1:blk
        inti = (ms(i)+1):ms(i+1);
        Z(inti, inti) = Z(inti, inti)';
        for j = i+1:blk
            intj = (ms(j)+1):ms(j+1);
            A = Z(inti, intj);
            Z(inti, intj) = Z(intj,inti)';
            Z(intj, inti) = A';
        end
    end

    inplace_chol(Z, 0, 1.0/m, lambda);

    dA = diag(Z);

    cholR = @(x) double(tri_solve_d(Z, 1, 1, dR, x));
    cholRt = @(x) double(tri_solve_d(Z, 1, 0, dR, x));
    cholA = @(x) double(tri_solve_d(Z, 0, 0, dA, x));
    cholAt = @(x) double(tri_solve_d(Z, 0, 1, dA, x));
end

function [download, upload] = produceDU(useGPU, isXsparse)
    if useGPU && ~isXsparse
        upload = @(x) gpuArray(x);
        download = @(x) gather(x);
    else
        upload = @(x) x;
        download = @(x) x;
    end
end

function p = KtKprod(Kr, u, z)

    bt = (numel(u)>0) + 2*(numel(z) > 0);

    if bt == 0
        p = 0;
        return;
    end

    switch bt
	    case 1
            u = Kr*u;
	    	p = (u'*Kr)';
	    case 2
	    	p = (z'*Kr)';
	    case 3
            z = z + Kr*u;
	    	p = (z'*Kr)';
    end
end


function p = KnmProd(X, C, u, z, blk, kern)

    if isdistributed(X)
        if ~isempty(z)
            z = distributed(z);
        end

        spmd
            p0 = fncKnmProd(getLocalPart(X), C, u, getLocalPart(z), blk, kern);
        end

        m = size(C,1);
        p = zeros(m,max(size(u,2), size(z,2)), 'like', C);
        for i=1:numel(p0)
            p = p + p0{i};
        end
    else
        p = fncKnmProd(X, C, u, z, blk, kern);
    end
end

function p = fncKnmProd(X, C, u, z, blk, kern)
    n = size(X,1); m = size(C,1);
    ms = ceil(linspace(0, n, blk+1));
    p = zeros(m,max(size(u,2), size(z,2)), 'like', C);

    zi = [];

    for i=1:blk

        X1 = feval(class(C), X((ms(i)+1):ms(i+1), :));

	    if ~isempty(z)
            zi  = z((ms(i)+1):ms(i+1), :);
        end

        clear Kr;
        Kr = kern(X1, C);

        p = p + KtKprod(Kr, u, zi);

	    fprintf('*');
    end
    clear Kr;
    clear X;
    clear X1;
    clear C;

    fprintf('\n');
end

function z = multW(KnmP, lambda, z, n, cholR, cholRt, cholA, cholAt)
    z = cholA(z);
    z = cholAt(cholRt(KnmP(cholR(z), [])/n) + lambda*z);
end

function x = conjgrad(funA, b, T, x, cobj, callback)

    m = size(b,1); dim = size(b,2);

    if ~exist('x','var') || numel(x) == 0
        r = b;
        x = zeros(m, dim);
    else
        r = double(b - funA(x));
    end
    cobj = callback(x, cobj);

    p = r;
    rsold = sum(r.*r, 1);

    for i = 1:T
        Ap = funA(p);
        alpha = rsold ./ sum(p .* Ap, 1);
        x = x + bsxfun(@times, alpha, p);

        cobj = callback(x, cobj);

        if mod(i, 20) == 0
            r = double(b - funA(x));
        else
            r = r - bsxfun(@times, alpha, Ap);
        end

        rsnew = sum(r.*r, 1);

        if prod(sqrt(rsnew) < 1e-10)
              break;
        end

        p = r + bsxfun(@times, rsnew ./ rsold, p);
        rsold = rsnew;
    end
end


function M = blockKernComp(A, B, kern, kerMemCoefficient, freeDoubles, useGPU)
    if numel(B) == 0
        M = blockKernCompGPUSymmetric(A, kern, kerMemCoefficient, freeDoubles, useGPU);
        return;
    end

    if useGPU && ~issparse(A) && ~issparse(B)
        gD = gpuDevice();
        dbmem = gD.AvailableMemory/8;
        d = 2*size(A,2);
        nmax_gpu = floor((sqrt(d^2 + kerMemCoefficient*dbmem)-d)/kerMemCoefficient);
        nmax_ram = floor(sqrt(freeDoubles));
        nmax = min(nmax_gpu, nmax_ram);

        if nmax > size(A,1) && size(A,1) <= size(B,1)
            nmaxA = size(A,1);
            nmaxB = min(floor((dbmem - nmaxA*(kerMemCoefficient + d))/(nmaxA + kerMemCoefficient + d)), floor(freeDoubles/nmaxA));
        elseif nmax > size(B,1)
            nmaxB = size(B,1);
            nmaxA = min(floor((dbmem - nmaxB*(kerMemCoefficient + d))/(nmaxB + kerMemCoefficient + d)), floor(freeDoubles/nmaxB));
        else
            nmaxA = nmax;
            nmaxB = nmax;
        end
    else
        nmax = floor(sqrt(freeDoubles));

        if nmax > size(A,1) && size(A,1) <= size(B,1)
            nmaxA = size(A,1);
            nmaxB = freeDoubles/(kerMemCoefficient*nmaxA);
        elseif nmax > size(B,1)
            nmaxB = size(B,1);
            nmaxA = freeDoubles/(kerMemCoefficient*nmaxB);
        else
            nmaxA = nmax;
            nmaxB = nmax;
        end
    end

    [download, upload] = produceDU(useGPU, issparse(A) | issparse(B));

    blkA = ceil(size(A,1)/nmaxA);
    as = ceil(linspace(0, size(A,1), blkA + 1));


    blkB = ceil(size(B,1)/nmaxB);
    bs = ceil(linspace(0, size(B,1), blkB + 1));

    if blkA == 1 && blkB == 1
        M = download(kern(upload(A), upload(B)));
    else

        M = zeros(size(A,1), size(B,1));

        for i=1:blkA
            C1 = upload(A(as(i)+1:as(i+1),:));
            for j = 1:blkB
                C2 = upload(B(bs(j)+1:bs(j+1), :));
                M(as(i)+1:as(i+1), bs(j)+1:bs(j+1)) = download(kern(C1,C2));
             end
        end
    end
end

function M = blockKernCompGPUSymmetric(A, kern, kerMemCoefficient, freeDoubles, useGPU)

    if useGPU && ~issparse(A)
        gD = gpuDevice();
        dbmem = gD.AvailableMemory/8;
        d = size(A,2);
        nmax_gpu = floor((sqrt(d^2 + kerMemCoefficient*dbmem)-d)/kerMemCoefficient);
        nmax_ram = floor(sqrt(freeDoubles));
        nmax = min(nmax_gpu, nmax_ram);
    else
        nmax = sqrt(freeDoubles/kerMemCoefficient);
    end

    [download, upload] = produceDU(useGPU, issparse(A));

    blkA = ceil(size(A,1)/nmax);
    as = ceil(linspace(0, size(A,1), blkA + 1));

    blkB = blkA;
    bs = as;

    if blkA == 1
        uA = upload(A);
        M = download(kern(uA, uA));
    else

        M = zeros(size(A,1), size(A,1));

        for i=1:blkA
            C1 = upload(A(as(i)+1:as(i+1),:));
            M(as(i)+1:as(i+1), as(i)+1:as(i+1)) = download(kern(C1,C1));
            for j = i+1:blkB
                C2 = upload(A(bs(j)+1:bs(j+1), :));
                M(as(i)+1:as(i+1), bs(j)+1:bs(j+1)) = download(kern(C1,C2));
             end
        end

        for i=1:blkA
            for j = i+1:blkB
                M(bs(j)+1:bs(j+1), as(i)+1:as(i+1)) = M(as(i)+1:as(i+1), bs(j)+1:bs(j+1))';
             end
        end

    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function memStats = memoryInfo()
    bytes2kBytes = @(bytes) round(bytes/1024);

    memStats = struct('total', false, 'free', false, 'used', false);
    if isunix

        [~, memUsageStr] = unix('free -k');
        try
            rr = strsplit(memUsageStr, '\n');
            memUsage = cell2mat(textscan(rr{2},'%*s %u64 %u64 %u64 %*u64 %*u64 %u64','delimiter',' ','collectoutput',true,'multipleDelimsAsOne',true));
            memStats.total = memUsage(1);
            memStats.used = memUsage(2);
            memStats.free = memUsage(3);
        catch err %#ok
        end

    else
        [~, sys] = memory;
        memStats.total = bytes2kBytes(sys.PhysicalMemory.Total);
        memStats.free = bytes2kBytes(sys.PhysicalMemory.Available);
        memStats.used = memStats.total - memStats.free;
    end

end
