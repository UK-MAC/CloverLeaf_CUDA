
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

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

    if(row >= (y_min + 1) && row <= (y_max + 1) + 1
    && column >= (x_min + 1) && column <= (x_max + 1) + 1)
    {
        xvel0[THARR2D(0, 0, 1)] = xvel1[THARR2D(0, 0, 1)];
        yvel0[THARR2D(0, 0, 1)] = yvel1[THARR2D(0, 0, 1)];

        if(row <= (y_max + 1)
        && column <= (x_max + 1))
        {
            density0[THARR2D(0, 0, 0)] = density1[THARR2D(0, 0, 0)];
            energy0[THARR2D(0, 0, 0)]  = energy1[THARR2D(0, 0, 0)];
        }
    }
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
    chunk.reset_field_kernel();
}

void CloverleafCudaChunk::reset_field_kernel
(void)
{
_CUDA_BEGIN_PROFILE_name(device);
    device_reset_field_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0, density1,
        energy0, energy1, xvel0, xvel1, yvel0, yvel1);
    errChk(__LINE__, __FILE__);
_CUDA_END_PROFILE_name(device);
}
