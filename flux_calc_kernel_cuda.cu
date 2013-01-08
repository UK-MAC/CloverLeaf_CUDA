
#include "cuda_common.cu"
#include "ftocmacros.h"
#include <iostream>
#include "omp.h"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

CudaDevPtrStorage pointer_storage;

__global__ void device_flux_calc_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
double dt,
const double * __restrict const xarea,
const double * __restrict const yarea,
const double * __restrict const xvel0,
const double * __restrict const yvel0,
const double * __restrict const xvel1,
const double * __restrict const yvel1,
      double * __restrict const vol_flux_x,
      double * __restrict const vol_flux_y)
{
    __kernel_indexes;

    if(column > 1 && column < x_max+3
    && row > 1 && row < y_max+2)
    {
        vol_flux_x[THARR2D(0, 0, 1)] = 0.25 * dt * xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]);
    }

    if(column > 1 && column < x_max+2
    && row > 1 && row < y_max+3)
    {
        vol_flux_y[THARR2D(0, 0, 0)] = 0.25 * dt * yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]);
    }

}

void flux_calc_cuda
(int x_min,int x_max,int y_min,int y_max,
double dt,
const double *xarea,
const double *yarea,
const double *xvel0,
const double *yvel0,
const double *xvel1,
const double *yvel1,
double * vol_flux_x,
double * vol_flux_y)
{

    double* xarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xarea, BUFSZ2D(1, 0));
    double* yarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yarea, BUFSZ2D(0, 1));

    double* xvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel0, BUFSZ2D(1, 1));
    double* xvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel1, BUFSZ2D(1, 1));
    double* yvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel0, BUFSZ2D(1, 1));
    double* yvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel1, BUFSZ2D(1, 1));

    double* vol_flux_x_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* vol_flux_y_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_flux_calc_kernel_cuda<<< ceil(((x_max+5)*(y_max+5))/static_cast<float>(BLOCK_SZ)), BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, xarea_d, yarea_d, xvel0_d, yvel0_d,
        xvel1_d, yvel1_d, vol_flux_x_d, vol_flux_y_d);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    pointer_storage.freeDevStorageAndCopy(vol_flux_x_d, vol_flux_x, BUFSZ2D(1, 0));
    pointer_storage.freeDevStorageAndCopy(vol_flux_y_d, vol_flux_y, BUFSZ2D(0, 1));

    pointer_storage.freeDevStorage(xarea_d);
    pointer_storage.freeDevStorage(yarea_d);
    pointer_storage.freeDevStorage(xvel0_d);
    pointer_storage.freeDevStorage(yvel0_d);
    pointer_storage.freeDevStorage(xvel1_d);
    pointer_storage.freeDevStorage(yvel1_d);

}

extern "C" void flux_calc_kernel_cuda_
(int *xmin,int *xmax,int *ymin,int *ymax,
double *dbyt,
const double *xarea,
const double *yarea,
const double *xvel0,
const double *yvel0,
const double *xvel1,
const double *yvel1,
double *vol_flux_x,
double *vol_flux_y)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    flux_calc_cuda(*xmin,*xmax,*ymin,*ymax, *dbyt, xarea, yarea,
        xvel0, yvel0, xvel1, yvel1, vol_flux_x, vol_flux_y);
    #else
    chunk.flux_calc_kernel(*dbyt);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}


void CloverleafCudaChunk::flux_calc_kernel
(double dbyt)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_flux_calc_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dbyt, xarea, yarea, xvel0, yvel0,
        xvel1, yvel1, vol_flux_x, vol_flux_y);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}
