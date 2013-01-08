
#include <iostream>
#include "ftocmacros.h"
#include "omp.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_accelerate_kernel_cuda
(int x_min,int x_max,int y_min,int y_max, double dbyt,
const double* __restrict const xarea,
const double* __restrict const yarea,
const double* __restrict const volume,
const double* __restrict const density0,
const double* __restrict const pressure,
const double* __restrict const viscosity,
const double* __restrict const xvel0,
const double* __restrict const yvel0,
      double* __restrict const xvel1,
      double* __restrict const yvel1)
{
    __kernel_indexes;

    double nodal_mass, step_by_mass;

    // prevent writing to *vel1, then read from it, then write to it again
    double xvel_temp, yvel_temp;

    if(row > 1 && row < y_max+3
    && column > 1 && column < x_max+3)
    {
        nodal_mass = 0.25             
            * (density0[THARR2D(-1, -1, 0)] * volume[THARR2D(-1, -1, 0)]
            + density0[THARR2D(0, -1, 0)] * volume[THARR2D(0, -1, 0)]
            + density0[THARR2D(0, 0, 0)] * volume[THARR2D(0, 0, 0)]
            + density0[THARR2D(-1, 0, 0)] * volume[THARR2D(-1, 0, 0)]);

        step_by_mass = 0.5 * dbyt / nodal_mass;

        // x velocities
        xvel_temp = xvel0[THARR2D(0, 0, 1)] - step_by_mass
            * (xarea[THARR2D(0, 0, 1)] * (pressure[THARR2D(0, 0, 0)] - pressure[THARR2D(-1, 0, 0)])
            + xarea[THARR2D(0, -1, 1)] * (pressure[THARR2D(0, -1, 0)] - pressure[THARR2D(-1, -1, 0)]));

        xvel1[THARR2D(0, 0, 1)] = xvel_temp - step_by_mass
            * (xarea[THARR2D(0, 0, 1)] * (viscosity[THARR2D(0, 0, 0)] - viscosity[THARR2D(-1, 0, 0)])
            + xarea[THARR2D(0, -1, 1)] * (viscosity[THARR2D(0, -1, 0)] - viscosity[THARR2D(-1, -1, 0)]));

        // y velocities
        yvel_temp = yvel0[THARR2D(0, 0, 1)] - step_by_mass
            * (yarea[THARR2D(0, 0, 0)] * (pressure[THARR2D(0, 0, 0)] - pressure[THARR2D(0, -1, 0)])
            + yarea[THARR2D(-1, 0, 0)] * (pressure[THARR2D(-1, 0, 0)] - pressure[THARR2D(-1, -1, 0)]));

        yvel1[THARR2D(0, 0, 1)] = yvel_temp - step_by_mass
            * (yarea[THARR2D(0, 0, 0)] * (viscosity[THARR2D(0, 0, 0)] - viscosity[THARR2D(0, -1, 0)])
            + yarea[THARR2D(-1, 0, 0)] * (viscosity[THARR2D(-1, 0, 0)] - viscosity[THARR2D(-1, -1, 0)]));

    }
    
}

void accelerate_cuda
(int x_min,int x_max,int y_min,int y_max,
double dbyt,
double *xarea,
double *yarea,
double *volume,
double *density0,
double *pressure,
double *viscosity,
double *xvel0,
double *yvel0,
double *xvel1,
double *yvel1)
{

    pointer_storage.setSize(x_max, y_max);

    double* volume_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, volume, BUFSZ2D(0, 0));
    double* pressure_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, pressure, BUFSZ2D(0, 0));
    double* viscosity_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, viscosity, BUFSZ2D(0, 0));
    double* density0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density0, BUFSZ2D(0, 0));

    double* xarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xarea, BUFSZ2D(1, 0));
    double* yarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yarea, BUFSZ2D(0, 1));

    double* xvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel0, BUFSZ2D(1, 1));
    double* yvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel0, BUFSZ2D(1, 1));

    double* xvel1_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* yvel1_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

    device_accelerate_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, dbyt, xarea_d, yarea_d, volume_d, density0_d,
        pressure_d, viscosity_d, xvel0_d, yvel0_d, xvel1_d, yvel1_d);

errChk(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    pointer_storage.freeDevStorageAndCopy(xvel1_d, xvel1, BUFSZ2D(1, 1));
    pointer_storage.freeDevStorageAndCopy(yvel1_d, yvel1, BUFSZ2D(1, 1));

    pointer_storage.freeDevStorage(volume_d);
    pointer_storage.freeDevStorage(density0_d);
    pointer_storage.freeDevStorage(pressure_d);
    pointer_storage.freeDevStorage(viscosity_d);

    pointer_storage.freeDevStorage(xarea_d);
    pointer_storage.freeDevStorage(yarea_d);
    pointer_storage.freeDevStorage(xvel0_d);
    pointer_storage.freeDevStorage(yvel0_d);
}

extern "C" void accelerate_kernel_cuda_
(int *xmin,int *xmax,int *ymin,int *ymax,
double *dbyt,
double *xarea,double *yarea,
double *volume,
double *density0,
double *pressure,
double *viscosity,
double *xvel0,
double *yvel0,
double *xvel1,
double *yvel1)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    accelerate_cuda(*xmin, *xmax, *ymin, *ymax, *dbyt, xarea, yarea, volume,
        density0, pressure, viscosity, xvel0, yvel0, xvel1, yvel1);
    #else
    chunk.accelerate_kernel(*dbyt);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::accelerate_kernel
(double dbyt)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

    device_accelerate_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, dbyt, xarea, yarea, volume, density0,
        pressure, viscosity, xvel0, yvel0, xvel1, yvel1);
    errChk(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}
