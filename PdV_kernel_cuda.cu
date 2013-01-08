
#include <iostream>
#include "cuda_common.cu"
#include "ftocmacros.h"
#include <algorithm>

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

#include "omp.h"

extern CudaDevPtrStorage pointer_storage;

__global__ void device_PdV_cuda_kernel_predict
(int x_min, int x_max, int y_min, int y_max, 
double dt,
         int * __restrict const error_condition,
const double * __restrict const xarea,
const double * __restrict const yarea,
const double * __restrict const volume,
const double * __restrict const density0,
      double * __restrict const density1,
const double * __restrict const energy0,
      double * __restrict const energy1,
const double * __restrict const pressure,
const double * __restrict const viscosity,
const double * __restrict const xvel0,
const double * __restrict const yvel0,
const double * __restrict const xvel1,
const double * __restrict const yvel1)
{
    __kernel_indexes;

    __shared__ int err_cond_kernel[BLOCK_SZ];
    err_cond_kernel[threadIdx.x] = 0;

    double volume_change;
    double recip_volume, energy_change, min_cell_volume,
        right_flux, left_flux, top_flux, bottom_flux, total_flux;
    
    if(row > 1 && column > 1
    && row < y_max+2 && column < x_max+2)
    {
        left_flux   = (xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 0, 1)] 
            + xvel0[THARR2D(0, 1, 1)] + xvel0[THARR2D(0, 1, 1)]))
            * 0.25 * dt * 0.5;
        right_flux  = (xarea[THARR2D(1, 0, 1)]
            * (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 0, 1)] 
            + xvel0[THARR2D(1, 1, 1)] + xvel0[THARR2D(1, 1, 1)]))
            * 0.25 * dt * 0.5;

        bottom_flux = (yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(0, 0, 1)] 
            + yvel0[THARR2D(1, 0, 1)] + yvel0[THARR2D(1, 0, 1)]))
            * 0.25 * dt * 0.5;
        top_flux    = (yarea[THARR2D(0, 1, 0)]
            * (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(0, 1, 1)] 
            + yvel0[THARR2D(1, 1, 1)] + yvel0[THARR2D(1, 1, 1)]))
            * 0.25 * dt * 0.5;

        total_flux = right_flux - left_flux + top_flux - bottom_flux;

        volume_change = volume[THARR2D(0, 0, 0)]
            / (volume[THARR2D(0, 0, 0)] + total_flux);

        //minimum of total, horizontal, and vertical flux
        min_cell_volume = 
            MIN(volume[THARR2D(0, 0, 0)] + total_flux,
            MIN(volume[THARR2D(0, 0, 0)] + right_flux - left_flux,
                volume[THARR2D(0, 0, 0)] + top_flux - bottom_flux));

        if(volume_change <= 0.0)
        {
            err_cond_kernel[threadIdx.x] = 1;
        }
        if(min_cell_volume <= 0.0)
        {
            err_cond_kernel[threadIdx.x] = 2;
        }

        recip_volume = 1.0/volume[THARR2D(0, 0, 0)];

        energy_change = ((pressure[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)])
            + (viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)]))
            * total_flux * recip_volume;

        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)] - energy_change;
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)] * volume_change;
    }

    //reduction to get error conditon, if any
    __syncthreads();
    for(size_t offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if(threadIdx.x < offset)
        {
            err_cond_kernel[threadIdx.x] = MAX(err_cond_kernel[threadIdx.x],
                err_cond_kernel[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    error_condition[blockIdx.x] = err_cond_kernel[0];;
}

__global__ void device_PdV_cuda_kernel_not_predict
(int x_min, int x_max, int y_min, int y_max, 
double dt,
         int * __restrict const error_condition,
const double * __restrict const xarea,
const double * __restrict const yarea,
const double * __restrict const volume,
const double * __restrict const density0,
      double * __restrict const density1,
const double * __restrict const energy0,
      double * __restrict const energy1,
const double * __restrict const pressure,
const double * __restrict const viscosity,
const double * __restrict const xvel0,
const double * __restrict const yvel0,
const double * __restrict const xvel1,
const double * __restrict const yvel1)
{
    __kernel_indexes;

    __shared__ int err_cond_kernel[BLOCK_SZ];
    err_cond_kernel[threadIdx.x] = 0;

    double volume_change;
    double recip_volume, energy_change, min_cell_volume,
        right_flux, left_flux, top_flux, bottom_flux, total_flux;
    
    if(row > 1 && column > 1
    && row < y_max+2 && column < x_max+2)
    {
        left_flux   = (xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 0, 1)] 
            + xvel0[THARR2D(0, 1, 1)] + xvel1[THARR2D(0, 1, 1)]))
            * 0.25 * dt;
        right_flux  = (xarea[THARR2D(1, 0, 1)]
            * (xvel0[THARR2D(1, 0, 1)] + xvel1[THARR2D(1, 0, 1)] 
            + xvel0[THARR2D(1, 1, 1)] + xvel1[THARR2D(1, 1, 1)]))
            * 0.25 * dt;

        bottom_flux = (yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel1[THARR2D(0, 0, 1)] 
            + yvel0[THARR2D(1, 0, 1)] + yvel1[THARR2D(1, 0, 1)]))
            * 0.25 * dt;
        top_flux    = (yarea[THARR2D(0, 1, 0)]
            * (yvel0[THARR2D(0, 1, 1)] + yvel1[THARR2D(0, 1, 1)] 
            + yvel0[THARR2D(1, 1, 1)] + yvel1[THARR2D(1, 1, 1)]))
            * 0.25 * dt;

        total_flux = right_flux - left_flux + top_flux - bottom_flux;

        volume_change = volume[THARR2D(0, 0, 0)]
            / (volume[THARR2D(0, 0, 0)] + total_flux);

        min_cell_volume =
            MIN(volume[THARR2D(0, 0, 0)] + total_flux,
            MIN(volume[THARR2D(0, 0, 0)] + right_flux - left_flux,
                volume[THARR2D(0, 0, 0)] + top_flux - bottom_flux));

        if(volume_change <= 0.0)
        {
            err_cond_kernel[threadIdx.x] = 1;
        }
        if(min_cell_volume <= 0.0)
        {
            err_cond_kernel[threadIdx.x] = 2;
        }

        recip_volume = 1.0/volume[THARR2D(0, 0, 0)];

        energy_change = ((pressure[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)])
            + (viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)]))
            * total_flux * recip_volume;

        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)] - energy_change;
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)] * volume_change;

    }

    __syncthreads();
    for(size_t offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if(threadIdx.x < offset)
        {
            err_cond_kernel[threadIdx.x] = MAX(err_cond_kernel[threadIdx.x],
                err_cond_kernel[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    error_condition[blockIdx.x] = err_cond_kernel[0];;
}

void PdV_cuda
(int error_condition,int predict,int x_min,int x_max,int y_min,int y_max,
double dt,
double *xarea,
double *yarea,
double *volume,
double *density0,
double *density1,
double *energy0,
double *energy1,
double *pressure,
double *viscosity,
double *xvel0,
double *yvel0,
double *xvel1,
double *yvel1)
{
    pointer_storage.setSize(x_max, y_max);

    double* xarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xarea, BUFSZ2D(1, 0));
    double* yarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yarea, BUFSZ2D(0, 1));

    double* density0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density0, BUFSZ2D(0, 0));
    double* energy0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, energy0, BUFSZ2D(0, 0));

    double* volume_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, volume, BUFSZ2D(0, 0));
    double* pressure_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, pressure, BUFSZ2D(0, 0));
    double* viscosity_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, viscosity, BUFSZ2D(0, 0));

    double* xvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel0, BUFSZ2D(1, 1));
    double* xvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel1, BUFSZ2D(1, 1));
    double* yvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel0, BUFSZ2D(1, 1));
    double* yvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel1, BUFSZ2D(1, 1));

    double* energy1_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* density1_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    size_t num_blocks = (((x_max+4)*(y_max+4))/BLOCK_SZ);
    //error condition
    thrust::device_ptr<int> reduce_ptr_1 =
        thrust::device_malloc<int>(num_blocks*sizeof(int));
    int* err_condition_arr_d = thrust::raw_pointer_cast(reduce_ptr_1);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

    if(predict)
    {
        device_PdV_cuda_kernel_predict<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, err_condition_arr_d,
            xarea_d, yarea_d, volume_d, density0_d, density1_d,
            energy0_d, energy1_d, pressure_d, viscosity_d,
            xvel0_d, yvel0_d, xvel1_d, yvel1_d);
    }
    else
    {
        device_PdV_cuda_kernel_not_predict<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, err_condition_arr_d,
            xarea_d, yarea_d, volume_d, density0_d, density1_d,
            energy0_d, energy1_d, pressure_d, viscosity_d,
            xvel0_d, yvel0_d, xvel1_d, yvel1_d);
    }

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

errChk(__LINE__, __FILE__);

    pointer_storage.freeDevStorageAndCopy(energy1_d, energy1, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(density1_d, density1, BUFSZ2D(0, 0));

    pointer_storage.freeDevStorage(xarea_d);
    pointer_storage.freeDevStorage(yarea_d);
    pointer_storage.freeDevStorage(volume_d);
    pointer_storage.freeDevStorage(pressure_d);
    pointer_storage.freeDevStorage(viscosity_d);
    pointer_storage.freeDevStorage(xvel0_d);
    pointer_storage.freeDevStorage(yvel0_d);
    pointer_storage.freeDevStorage(xvel1_d);
    pointer_storage.freeDevStorage(yvel1_d);
    pointer_storage.freeDevStorage(energy0_d);
    pointer_storage.freeDevStorage(density0_d);

    /*
    int err_cond = thrust::reduce(reduce_ptr_1,
        reduce_ptr_1 + num_blocks,
        0, thrust::maximum<int>());
    // */
    int err_cond = *thrust::max_element(reduce_ptr_1, reduce_ptr_1 + num_blocks);
    thrust::device_free(reduce_ptr_1);

    if(err_cond == 1)
    {
        std::cerr << "Negative volume in PdV kernel" << std::endl;
    }
    else if(err_cond == 2)
    {
        std::cerr << "Negative cell volume in PdV kernel" << std::endl;
    }

}

extern "C" void pdv_kernel_cuda_
(int *errorcondition,int *prdct,
int *xmin,int *xmax,int *ymin,int *ymax,double *dtbyt,
double *xarea,double *yarea,double *volume,
double *density0,
double *density1,
double *energy0,
double *energy1,
double *pressure,
double *viscosity,
double *xvel0,
double *xvel1,
double *yvel0,
double *yvel1)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    PdV_cuda(*errorcondition, *prdct,  *xmin, *xmax, *ymin, *ymax,*dtbyt,
        xarea, yarea, volume, density0, density1, energy0, energy1,
        pressure, viscosity, xvel0, yvel0, xvel1, yvel1);
    #else
    chunk.PdV_kernel(errorcondition, *prdct, *dtbyt);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}


void CloverleafCudaChunk::PdV_kernel
(int* error_condition, int predict, double dt)
{

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

    if(predict)
    {
        device_PdV_cuda_kernel_predict<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, pdv_reduce_array,
            xarea, yarea, volume, density0, density1,
            energy0, energy1, pressure, viscosity,
            xvel0, yvel0, xvel1, yvel1);
        errChk(__LINE__, __FILE__);
    }
    else
    {
        device_PdV_cuda_kernel_not_predict<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, pdv_reduce_array,
            xarea, yarea, volume, density0, density1,
            energy0, energy1, pressure, viscosity,
            xvel0, yvel0, xvel1, yvel1);
        errChk(__LINE__, __FILE__);
    }

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    int err_cond = *thrust::max_element(reduce_pdv,
        reduce_pdv + num_blocks);

    if(err_cond == 1)
    {
        std::cerr << "Negative volume in PdV kernel" << std::endl;
    }
    else if(err_cond == 2)
    {
        std::cerr << "Negative cell volume in PdV kernel" << std::endl;
    }
}

