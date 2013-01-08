
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_initialise_chunk_kernel_vertex_cuda
(int x_min,int x_max,int y_min,int y_max,
double d_xmin,
double d_ymin,
double d_dx,
double d_dy,

double* __restrict const vertexx,
double* __restrict const vertexdx,
double* __restrict const vertexy,
double* __restrict const vertexdy)
{
    const int glob_id = threadIdx.x
        + blockIdx.x * blockDim.x;

    //bigger indexes 
    const int row = glob_id / (x_max + 5);
    const int column = glob_id % (x_max + 5);

    //fill out x arrays
    if(column < x_max+5)
    {
        vertexx[column] = d_xmin + d_dx * (double)((column - 1) - x_min);
        vertexdx[column] = d_dx;
    }

    // fill out y arrays
    if(row < y_max+5)
    {
        vertexy[row] = d_ymin + d_dy * (double)((row - 1) - y_min);
        vertexdy[row] = d_dy;
    }

}

__global__ void device_initialise_chunk_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
double d_xmin,
double d_ymin,
double d_dx,
double d_dy,

const double* __restrict const vertexx,
const double* __restrict const vertexdx,
const double* __restrict const vertexy,
const double* __restrict const vertexdy,
      double* __restrict const cellx,
      double* __restrict const celldx,
      double* __restrict const celly,
      double* __restrict const celldy,

double* __restrict const volume, 
double* __restrict const xarea, 
double* __restrict const yarea)
{
    __kernel_indexes;

    //fill x arrays
    if(column < x_max+4)
    {
        cellx[column] = 0.5 * (vertexx[column] + vertexx[column + 1]);
        celldx[column] = d_dx;
    }

    //fill y arrays
    if(row < y_max+4)
    {
        celly[row] = 0.5 * (vertexy[row] + vertexy[row + 1]);
        celldy[row] = d_dy;
    }

    if(row < y_max+4
    && column < x_max+4)
    {
        volume[THARR2D(0, 0, 0)] = d_dx * d_dy;
        xarea[THARR2D(0, 0, 1)] = celldy[row];
        yarea[THARR2D(0, 0, 0)] = celldx[column];
    }

}

void initialise_chunk_cuda
(int x_min,int x_max,int y_min,int y_max,
double d_xmin,
double d_ymin,
double d_dx,
double d_dy,

double* vertexx,
double* vertexdx,
double* vertexy,
double* vertexdy,
double* cellx,
double* celldx,
double* celly,
double* celldy,

double* volume, 
double* xarea, 
double* yarea)
{

    pointer_storage.setSize(x_max, y_max);

    double* volume_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* xarea_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* yarea_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    double* vertexx_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* vertexdx_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* vertexy_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* vertexdy_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    double* cellx_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* celldx_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* celly_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* celldy_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    //bigger indexes to allow for filling the very last bit of the vertex arrays
    device_initialise_chunk_kernel_vertex_cuda<<< ((x_max+5)*(y_max+5))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx_d, vertexdx_d, vertexy_d, vertexdy_d);
    errChk(__LINE__, __FILE__);

    device_initialise_chunk_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx_d, vertexdx_d, vertexy_d, vertexdy_d,
        cellx_d, celldx_d, celly_d, celldy_d,
        volume_d, xarea_d, yarea_d);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    pointer_storage.freeDevStorageAndCopy(volume_d, volume, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(xarea_d, xarea, BUFSZ2D(1, 0));
    pointer_storage.freeDevStorageAndCopy(yarea_d, yarea, BUFSZ2D(0, 1));

    pointer_storage.freeDevStorageAndCopy(vertexx_d, vertexx, (x_max+5)*sizeof(double));
    pointer_storage.freeDevStorageAndCopy(vertexdx_d, vertexdx, (x_max+5)*sizeof(double));
    pointer_storage.freeDevStorageAndCopy(vertexy_d, vertexy, (y_max+5)*sizeof(double));
    pointer_storage.freeDevStorageAndCopy(vertexdy_d, vertexdy, (y_max+5)*sizeof(double));

    pointer_storage.freeDevStorageAndCopy(cellx_d, cellx, (x_max+4)*sizeof(double));
    pointer_storage.freeDevStorageAndCopy(celldx_d, celldx, (x_max+4)*sizeof(double));
    pointer_storage.freeDevStorageAndCopy(celly_d, celly, (y_max+4)*sizeof(double));
    pointer_storage.freeDevStorageAndCopy(celldy_d, celldy, (y_max+4)*sizeof(double));

}

void initialise_chunk_kernel_
(int *x_min,int *x_max,int *y_min,int *y_max,
double* d_xmin,
double* d_ymin,
double* d_dx,
double* d_dy,

double* vertexx,
double* vertexdx,
double* vertexy,
double* vertexdy,
double* cellx,
double* celldx,
double* celly,
double* celldy,

double* volume, 
double* xarea, 
double* yarea);

extern "C" void initialise_chunk_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
double* d_xmin,
double* d_ymin,
double* d_dx,
double* d_dy,

double* vertexx,
double* vertexdx,
double* vertexy,
double* vertexdy,
double* cellx,
double* celldx,
double* celly,
double* celldy,

double* volume, 
double* xarea, 
double* yarea)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    initialise_chunk_cuda(*x_min, *x_max, *y_min, *y_max,
        *d_xmin, *d_ymin, *d_dx, *d_dy,
        vertexx, vertexdx, vertexy, vertexdy,
        cellx, celldx, celly, celldy,
        volume, xarea, yarea);
    #else
    chunk.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    //bigger indexes to allow for filling the very last bit of the vertex arrays
    device_initialise_chunk_kernel_vertex_cuda<<< ((x_max+5)*(y_max+5))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy);
    errChk(__LINE__, __FILE__);

    device_initialise_chunk_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy,
        cellx, celldx, celly, celldy,
        volume, xarea, yarea);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}
