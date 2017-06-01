#include "mex.h"
#include "lapack.h"

template<class DTYPE>
void addMult(DTYPE *A, mwSignedIndex m, DTYPE p, DTYPE q, int up){
    mwSignedIndex i, j, k;
    
    if(p != 1.0){
        if(up == 0){
            k = 0;
            for(i=0; i < m; i++){
                for (j=0; j <= i; j++)
                    A[k++] *= p;
                k += m - i - 1;
            }
        }else{
            k = 0;
            for(i=0; i < m; i++){
                k += i;
                for (j=0; j < m-i; j++)
                    A[k++] *= p;
            }
        }
    }
    
    if(q != 0.0){
        for(i=0; i< m*m; i+=m+1)
            A[i] += q;
    }
}

void inplace_chol(const char *type, double *A, mwSignedIndex *m){
    mwSignedIndex info = 0;
    
    dpotrf(type, m, A, m, &info);
    
    if (info < 0) {
        mexErrMsgTxt("Parameters had an illegal value.");
    } else if (info > 0) {
        mexErrMsgTxt("Matrix is not positive-definite.");
    }
}

void inplace_chol(const char *type, float *A, mwSignedIndex *m){
    mwSignedIndex info = 0;
    
    spotrf(type, m, A, m, &info);
    
    if (info < 0) {
        mexErrMsgTxt("Parameters had an illegal value.");
    } else if (info > 0) {
        mexErrMsgTxt("Matrix is not positive-definite.");
    }
}

template<class DTYPE>
void fnc(mxArray *plhs[], const mxArray *prhs[]){
    int up = (int) mxGetScalar(prhs[1]);
    DTYPE p = mxGetScalar(prhs[2]); 
    DTYPE q = mxGetScalar(prhs[3]);
    
    /* pointer to data */
    DTYPE *A = (DTYPE *) mxGetData(prhs[0]);
    mwSignedIndex m = mxGetN(prhs[0]);
    
    addMult<DTYPE>(A, m, p, q, up);
    
    if(up == 0)
        inplace_chol("U", A, &m);
    else
        inplace_chol("L", A, &m);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    /* compile with: mex -largeArrayDims computeRA.c -lmwblas -lmwlapack */
    
    /* verify arguments */
    if (nrhs != 4 || nlhs != 0) {
        mexErrMsgTxt("Wrong number of arguments.");
    }
    if (!(mxIsDouble(prhs[0]) || mxIsSingle(prhs[0])) || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Input must be a real double matrix.");
    }
    if (mxGetM(prhs[0]) != mxGetN(prhs[0])) {
        mexErrMsgTxt("Input must be a symmetric positive-definite matrix.");
    }
    
    if(mxIsDouble(prhs[0]))
        fnc<double>(plhs, prhs);
    else
        fnc<float>(plhs, prhs);
}
    