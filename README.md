# Compilation

This is an example to make Cloverleaf on a Cray machine:

```
make COMPILER=CRAY NV_ARCH=FERMI -j 6 C_MPI_COMPILER=cc MPI_COMPILER=ftn 
```

* COMPILER is the same as with the other implementations.
* NV_ARCH is the architecture - FERMI or KEPLER only, must be specified because of memory limitations of shared memory when doing reductions.
* Job number only affects parallel build of CUDA files
* (C_)MPI_COMPILER affects which compiler to use

## Device selection

To pay attention to the `cuda_device` variable in clover.in pass `-D
MANUALLY_CHOOSE_GPU` into the `NV_OPTIONS` variable - ie:

`
make NV_OPTIONS="-DMANUALLY_CHOOSE_GPU" COMPILER=....
`

Device selection works differently on different machines, depending on the
number of GPUs in the system and whether they are set to exclusive mode - try
with and without if it doesn't work.
