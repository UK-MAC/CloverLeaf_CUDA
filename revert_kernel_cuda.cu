
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_revert_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* __restrict const density0,
      double* __restrict const density1,
const double* __restrict const energy0,
      double* __restrict const energy1)
{
    __kernel_indexes;

    if(row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)];
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)];
    }
}

extern "C" void revert_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
const double* density0,
      double* density1,
const double* energy0,
      double* energy1)
{
    chunk.revert_kernel();
}

void CloverleafCudaChunk::revert_kernel
(void)
{
_CUDA_BEGIN_PROFILE_name(device);
    device_revert_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0, density1, energy0, energy1);
    errChk(__LINE__, __FILE__);
_CUDA_END_PROFILE_name(device);
}

