#include "mex.h"
#include "blas.h"


void trsm(const char *side, const char *uplo, const char *transa, const char *diag, const ptrdiff_t *m,
    const ptrdiff_t *n, const double *alpha, const double *a, const ptrdiff_t *lda, double *b, const ptrdiff_t *ldb){
    dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void trsm(const char *side, const char *uplo, const char *transa, const char *diag, const ptrdiff_t *m,
    const ptrdiff_t *n, const float *alpha, const float *a, const ptrdiff_t *lda, float *b, const ptrdiff_t *ldb){
    strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

template<class DTYPE> 
void tri_solve(mxArray *plhs[], const mxArray *prhs[]){
    DTYPE *Z = (DTYPE *) mxGetData(prhs[0]);
    int lo = mxGetScalar(prhs[1]) > 0.5;
    int tr = mxGetScalar(prhs[2]) > 0.5;
    DTYPE *d = (DTYPE *) mxGetData(prhs[3]);
    
    plhs[0] = mxDuplicateArray(prhs[4]);
    DTYPE *res = (DTYPE *) mxGetData(plhs[0]);
    
    mwSignedIndex m = mxGetN(prhs[0]);
    mwSignedIndex dim = mxGetN(prhs[4]);
    
    mwSignedIndex i, k = 0;
    for(i=0; i< m*m; i+=m+1)
            Z[i] = d[k++];
    
    const DTYPE alpha = 1.0;
    
    if(!lo && !tr){
        trsm("L", "U", "N", "N", &m, &dim, &alpha, Z, &m, res, &m);
    }else if(!lo && tr){
        trsm("L", "U", "T", "N", &m, &dim, &alpha, Z, &m, res, &m);
    }else if(lo && !tr){
        trsm("L", "L", "N", "N", &m, &dim, &alpha, Z, &m, res, &m); 
    }else if(lo && tr){
        trsm("L", "L", "T", "N", &m, &dim, &alpha, Z, &m, res, &m);
    }
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    /* compile with: mex -largeArrayDims computeRA.c -lmwblas -lmwlapack */
    /*tri_solve_d(Z, 0, 0, d,  x) inplace solve (diag(d) + triu(Z,1))\x
     *tri_solve_d(Z, 0, 1, d, x) inplace solve (diag(d) + triu(Z,1))'\x
     *tri_solve_d(Z, 1, 0, d, x) inplace solve (diag(d) + tril(Z,1))\x
     *tri_solve_d(Z, 1, 1, d, x) inplace solve (diag(d) + tril(Z,1))'\x
    
    /* verify arguments */
    if (nrhs != 5 || nlhs != 1) {
        mexErrMsgTxt("Wrong number of arguments.");
    }
    if (!(mxIsDouble(prhs[0]) || mxIsSingle(prhs[0])) || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Input must be a real double or single matrix.");
    }
    if (!(mxIsDouble(prhs[3]) || mxIsSingle(prhs[3])) || mxIsComplex(prhs[3])) {
        mexErrMsgTxt("Input must be a real double or single matrix.");
    }
    if (!(mxIsDouble(prhs[4]) || mxIsSingle(prhs[4])) || mxIsComplex(prhs[4])) {
        mexErrMsgTxt("Input must be a real double or single matrix.");
    }
    if (mxIsDouble(prhs[0]) != mxIsDouble(prhs[3]) || mxIsDouble(prhs[3]) != mxIsDouble(prhs[4])) {
        mexErrMsgTxt("Z, d, x must be of the same datatype (real float, or real double).");
    }
    if (mxGetM(prhs[0]) != mxGetN(prhs[0])) {
        mexErrMsgTxt("Input must be a symmetric positive-definite matrix.");
    }
    
    if(mxIsDouble(prhs[0]))
        tri_solve<double>(plhs, prhs);
    else
        tri_solve<float>(plhs, prhs);
}

