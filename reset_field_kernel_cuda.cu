
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_reset_field_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
      double* __restrict const density0,
const double* __restrict const density1,
      double* __restrict const energy0,
const double* __restrict const energy1,
      double* __restrict const xvel0,
const double* __restrict const xvel1,
      double* __restrict const yvel0,
const double* __restrict const yvel1)
{
    __kernel_indexes;

    if(row > 1 && row < y_max+3
    && column > 1 && column < x_max+3)
    {
        xvel0[THARR2D(0, 0, 1)] = xvel1[THARR2D(0, 0, 1)];
        yvel0[THARR2D(0, 0, 1)] = yvel1[THARR2D(0, 0, 1)];

        if(row < y_max+2
        && column < x_max+2)
        {
            density0[THARR2D(0, 0, 0)] = density1[THARR2D(0, 0, 0)];
            energy0[THARR2D(0, 0, 0)]  = energy1[THARR2D(0, 0, 0)];
        }
    }
}

void reset_field_cuda
(int x_min,int x_max,int y_min,int y_max,
      double* density0,
const double* density1,
      double* energy0,
const double* energy1,
      double* xvel0,
const double* xvel1,
      double* yvel0,
const double* yvel1)
{

    pointer_storage.setSize(x_max, y_max);

    double* density1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density1, BUFSZ2D(0, 0));
    double* energy1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, energy1, BUFSZ2D(0, 0));
    double* xvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel1, BUFSZ2D(1, 1));
    double* yvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel1, BUFSZ2D(1, 1));

    double* density0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* energy0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* xvel0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* yvel0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_reset_field_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0_d, density1_d,
        energy0_d, energy1_d, xvel0_d, xvel1_d, yvel0_d, yvel1_d);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    pointer_storage.freeDevStorage(density1_d);
    pointer_storage.freeDevStorage(energy1_d);
    pointer_storage.freeDevStorage(xvel1_d);
    pointer_storage.freeDevStorage(yvel1_d);

    pointer_storage.freeDevStorageAndCopy(density0_d, density0, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(energy0_d, energy0, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(xvel0_d, xvel0, BUFSZ2D(1, 1));
    pointer_storage.freeDevStorageAndCopy(yvel0_d, yvel0, BUFSZ2D(1, 1));

}

extern "C" void reset_field_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
      double* density0,
const double* density1,
      double* energy0,
const double* energy1,
      double* xvel0,
const double* xvel1,
      double* yvel0,
const double* yvel1)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    reset_field_cuda(*x_min, *x_max, *y_min, *y_max, 
        density0, density1, energy0, energy1,
        xvel0, xvel1, yvel0, yvel1);
    #else
    chunk.reset_field_kernel();
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::reset_field_kernel
(void)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_reset_field_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0, density1,
        energy0, energy1, xvel0, xvel1, yvel0, yvel1);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}
