# Compilation

This is an example to make Cloverleaf on a Cray machine:

make COMPILER=CRAY NV_ARCH=FERMI -j 6 C_MPI_COMPILER=cc MPI_COMPILER=ftn 

* COMPILER is the same as with the other implementations.
* NV_ARCH is the architecture - FERMI or KEPLER only, must be specified because of memory limitations of shared memory when doing reductions.
* Job number only affects parallel build of CUDA files
* (C_)MPI_COMPILER affects which compiler to use

