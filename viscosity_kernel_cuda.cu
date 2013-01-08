
#include <iostream>
#include "ftocmacros.h"
#include "omp.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_viscosity_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double * __restrict const celldx,
const double * __restrict const celldy,
const double * __restrict const density0,
const double * __restrict const pressure,
      double * __restrict const viscosity,
const double * __restrict const xvel0,
const double * __restrict const yvel0)
{
    __kernel_indexes;

    double ugrad, vgrad, grad2, pgradx, pgrady, pgradx2, pgrady2,
        grad, ygrad, pgrad, xgrad, div, strain2, limiter;

    if(row > y_min && row < y_max+2
    && column > x_min && column < x_max+2)
    {
        ugrad = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])
            - (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]);

        vgrad = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])
            - (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]);
        
        div = (ugrad * celldx[column]) + (vgrad * celldy[row]);

        strain2 = 0.5 * (xvel0[THARR2D(0, 1, 1)] + xvel0[THARR2D(1, 1, 1)]
            - xvel0[THARR2D(0, 0, 1)] - xvel0[THARR2D(1, 0, 1)])/celldy[row]
            + 0.5 * (yvel0[THARR2D(1, 0, 1)] + yvel0[THARR2D(1, 1, 1)]
            - yvel0[THARR2D(0, 0, 1)] - yvel0[THARR2D(0, 1, 1)])/celldx[column];

        pgradx = (pressure[THARR2D(1, 0, 0)] - pressure[THARR2D(-1, 0, 0)])
            / (celldx[column] + celldx[column + 1]);
        pgrady = (pressure[THARR2D(0, 1, 0)] - pressure[THARR2D(0, -1, 0)])
            / (celldy[row] + celldy[row + 1]);

        pgradx2 = pgradx*pgradx;
        pgrady2 = pgrady*pgrady;

        limiter = ((0.5 * ugrad / celldx[column]) * pgradx2
            + ((0.5 * vgrad / celldy[row]) * pgrady2)
            + (strain2 * pgradx * pgrady))
            / MAX(pgradx2 + pgrady2, 1.0e-16);

        pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)),pgradx);
        pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)),pgrady);
        pgrad = sqrt((pgradx * pgradx) + (pgrady * pgrady));

        xgrad = fabs(celldx[column] * pgrad / pgradx);
        ygrad = fabs(celldy[row] * pgrad / pgrady);

        grad = MIN(xgrad, ygrad);
        grad2 = grad * grad;

        if(limiter > 0 || div >= 0.0)
        {
            viscosity[THARR2D(0,0,0)] = 0.0;
        }
        else
        {
            viscosity[THARR2D(0,0,0)] = 2.0 * density0[THARR2D(0,0,0)] * grad2 * (limiter * limiter);
        }
    }
}

void viscosity_cuda
(int x_min,int x_max,int y_min,int y_max,
const double *celldx,
const double *celldy,
const double *density0,
const double *pressure,
double *viscosity,
const double *xvel0,
const double *yvel0)
{

    pointer_storage.setSize(x_max, y_max);

    double* celldx_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celldx, (x_max+4)*sizeof(double));
    double* celldy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celldy, (y_max+4)*sizeof(double));
    double* viscosity_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    double* density0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density0, BUFSZ2D(0, 0));
    double* pressure_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, pressure, BUFSZ2D(0, 0));

    double* xvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel0, BUFSZ2D(1, 1));
    double* yvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel0, BUFSZ2D(1, 1));

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_viscosity_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, celldx_d, celldy_d, density0_d, pressure_d, viscosity_d, xvel0_d, yvel0_d);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

errChk(__LINE__, __FILE__);

    pointer_storage.freeDevStorageAndCopy(viscosity_d, viscosity, BUFSZ2D(0, 0));

    pointer_storage.freeDevStorage(celldx_d);
    pointer_storage.freeDevStorage(celldy_d);
    pointer_storage.freeDevStorage(density0_d);
    pointer_storage.freeDevStorage(pressure_d);
    pointer_storage.freeDevStorage(xvel0_d);
    pointer_storage.freeDevStorage(yvel0_d);
}

extern "C" void viscosity_kernel_cuda_
(int *xmin,int *x_max,int *ymin,int *y_max,
const double *celldx,
const double *celldy,
const double *density0,
const double *pressure,
double *viscosity,
const double *xvel0,
const double *yvel0)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    viscosity_cuda(*xmin, *x_max, *ymin, *y_max,celldx,celldy,density0,pressure,viscosity,xvel0,yvel0);
    #else
    chunk.viscosity_kernel();
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::viscosity_kernel
(void)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_viscosity_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, celldx, celldy, density0, pressure, viscosity, xvel0, yvel0);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

}
