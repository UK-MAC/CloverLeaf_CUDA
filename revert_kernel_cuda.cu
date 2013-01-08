
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_revert_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* __restrict const density0,
      double* __restrict const density1,
const double* __restrict const energy0,
      double* __restrict const energy1)
{
    __kernel_indexes;

    if(row > 1 && row < y_max+2
    && column > 1 && column < x_max+2)
    {
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)];
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)];
    }
}

void revert_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* density0,
      double* density1,
const double* energy0,
      double* energy1)
{

    pointer_storage.setSize(x_max, y_max);

    double* density0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density0, BUFSZ2D(0, 0));
    double* energy0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, energy0, BUFSZ2D(0, 0));

    double* density1_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* energy1_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_revert_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0_d, density1_d, energy0_d, energy1_d);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    pointer_storage.freeDevStorage(density0_d);
    pointer_storage.freeDevStorage(energy0_d);

    pointer_storage.freeDevStorageAndCopy(density1_d, density1, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(energy1_d, energy1, BUFSZ2D(0, 0));

}

extern "C" void revert_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
const double* density0,
      double* density1,
const double* energy0,
      double* energy1)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    revert_cuda(*x_min, *x_max, *y_min, *y_max, 
        density0, density1, energy0, energy1);
    #else
    chunk.revert_kernel();
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::revert_kernel
(void)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_revert_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0, density1, energy0, energy1);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}

