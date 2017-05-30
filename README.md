# FALKON_paper

Navigate in the matlab shell to the path containing FALKON's code

Run in the matlab shell:

mex -largeArrayDims ./tri_solve_d.cpp -lmwblas -lmwlapack

mex -largeArrayDims ./inplace_chol.cpp -lmwblas -lmwlapack
