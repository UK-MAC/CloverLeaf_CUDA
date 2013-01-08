
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_field_summary_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* __restrict const volume,
const double* __restrict const density0,
const double* __restrict const energy0,
const double* __restrict const pressure,
const double* __restrict const xvel0,
const double* __restrict const yvel0,

//output
double* __restrict const vol,
double* __restrict const mass,
double* __restrict const ie,
double* __restrict const ke,
double* __restrict const press)
{
    __kernel_indexes;

    __shared__ double vol_shared[BLOCK_SZ];
    __shared__ double mass_shared[BLOCK_SZ];
    __shared__ double ie_shared[BLOCK_SZ];
    __shared__ double ke_shared[BLOCK_SZ];
    __shared__ double press_shared[BLOCK_SZ];
    vol_shared[threadIdx.x] = 0.0;
    mass_shared[threadIdx.x] = 0.0;
    ie_shared[threadIdx.x] = 0.0;
    ke_shared[threadIdx.x] = 0.0;
    press_shared[threadIdx.x] = 0.0;

    if(row > 1 && row < y_max+2
    && column > 1 && column < x_max+2)
    {
        double vsqrd = 0.0;

        //unrolled do loop
        vsqrd += 0.25 * (xvel0[THARR2D(0, 0, 1)] * xvel0[THARR2D(0, 0, 1)]
                        +yvel0[THARR2D(0, 0, 1)] * yvel0[THARR2D(0, 0, 1)]);

        vsqrd += 0.25 * (xvel0[THARR2D(1, 0, 1)] * xvel0[THARR2D(1, 0, 1)]
                        +yvel0[THARR2D(1, 0, 1)] * yvel0[THARR2D(1, 0, 1)]);

        vsqrd += 0.25 * (xvel0[THARR2D(0, 1, 1)] * xvel0[THARR2D(0, 1, 1)]
                        +yvel0[THARR2D(0, 1, 1)] * yvel0[THARR2D(0, 1, 1)]);

        vsqrd += 0.25 * (xvel0[THARR2D(1, 1, 1)] * xvel0[THARR2D(1, 1, 1)]
                        +yvel0[THARR2D(1, 1, 1)] * yvel0[THARR2D(1, 1, 1)]);

        double cell_vol = volume[THARR2D(0, 0, 0)];
        double cell_mass = cell_vol * density0[THARR2D(0, 0, 0)];

        vol_shared[threadIdx.x] = cell_vol;
        mass_shared[threadIdx.x] = cell_mass;
        ie_shared[threadIdx.x] = cell_mass * energy0[THARR2D(0, 0, 0)];
        ke_shared[threadIdx.x] = cell_mass * 0.5 * vsqrd;
        press_shared[threadIdx.x] = cell_vol * pressure[THARR2D(0, 0, 0)];

    }

    __syncthreads();
    //*
    for(size_t offset = BLOCK_SZ / 2; offset > 0; offset /= 2)
    {
        if(threadIdx.x < offset)
        {
            vol_shared[threadIdx.x] += vol_shared[threadIdx.x + offset];
            mass_shared[threadIdx.x] += mass_shared[threadIdx.x + offset];
            ie_shared[threadIdx.x] += ie_shared[threadIdx.x + offset];
            ke_shared[threadIdx.x] += ke_shared[threadIdx.x + offset];
            press_shared[threadIdx.x] += press_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    // */

    vol[blockIdx.x] = vol_shared[0];
    mass[blockIdx.x] = mass_shared[0];
    ie[blockIdx.x] = ie_shared[0];
    ke[blockIdx.x] = ke_shared[0];
    press[blockIdx.x] = press_shared[0];

}

void field_summary_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* volume,
const double* density0,
const double* energy0,
const double* pressure,
const double* xvel0,
const double* yvel0,

//output
double* vol,
double* mass,
double* ie,
double* ke,
double* press)
{

    pointer_storage.setSize(x_max, y_max);

    double* volume_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, volume, BUFSZ2D(0, 0));
    double* density0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density0, BUFSZ2D(0, 0));
    double* energy0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, energy0, BUFSZ2D(0, 0));
    double* pressure_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, pressure, BUFSZ2D(0, 0));
    double* xvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel0, BUFSZ2D(1, 1));
    double* yvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel0, BUFSZ2D(1, 1));

    size_t num_blocks = (((x_max+4)*(y_max+4))/BLOCK_SZ);
    //vol
    thrust::device_ptr<double> thr_vol =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* thr_vol_d = thrust::raw_pointer_cast(thr_vol);
    //mass
    thrust::device_ptr<double> thr_mass =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* thr_mass_d = thrust::raw_pointer_cast(thr_mass);
    //ie
    thrust::device_ptr<double> thr_ie =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* thr_ie_d = thrust::raw_pointer_cast(thr_ie);
    //ke
    thrust::device_ptr<double> thr_ke =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* thr_ke_d = thrust::raw_pointer_cast(thr_ke);
    //press
    thrust::device_ptr<double> thr_press =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* thr_press_d = thrust::raw_pointer_cast(thr_press);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_field_summary_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, volume_d, density0_d,
        energy0_d, pressure_d, xvel0_d, yvel0_d,
        thr_vol_d, thr_mass_d, thr_ie_d, thr_ke_d, thr_press_d);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    *vol = thrust::reduce(thr_vol,
        thr_vol + num_blocks);

    *mass = thrust::reduce(thr_mass,
        thr_mass + num_blocks);

    *ie = thrust::reduce(thr_ie,
        thr_ie + num_blocks);

    *ke = thrust::reduce(thr_ke,
        thr_ke + num_blocks);

    *press = thrust::reduce(thr_press,
        thr_press + num_blocks);

    pointer_storage.freeDevStorage(volume_d);
    pointer_storage.freeDevStorage(density0_d);
    pointer_storage.freeDevStorage(energy0_d);
    pointer_storage.freeDevStorage(pressure_d);
    pointer_storage.freeDevStorage(xvel0_d);
    pointer_storage.freeDevStorage(yvel0_d);

    thrust::device_free(thr_vol);
    thrust::device_free(thr_mass);
    thrust::device_free(thr_ie);
    thrust::device_free(thr_ke);
    thrust::device_free(thr_press);

}

extern "C" void field_summary_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
const double* volume,
const double* density0,
const double* energy0,
const double* pressure,
const double* xvel0,
const double* yvel0,

//output
double* vol,
double* mass,
double* ie,
double* ke,
double* press)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    field_summary_cuda( *x_min, *x_max, *y_min, *y_max, volume, density0,
        energy0, pressure, xvel0, yvel0, vol, mass, ie, ke, press);
    #else
    chunk.field_summary_kernel(vol, mass, ie, ke, press);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::field_summary_kernel
(double* vol, double* mass,
double* ie, double* ke, double* press)
{
    
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_field_summary_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, volume, density0,
        energy0, pressure, xvel0, yvel0,
        work_array_1, work_array_2, work_array_3,
        work_array_4, work_array_5);
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    *vol = thrust::reduce(reduce_ptr_1,
        reduce_ptr_1 + num_blocks);

    *mass = thrust::reduce(reduce_ptr_2,
        reduce_ptr_2 + num_blocks);

    *ie = thrust::reduce(reduce_ptr_3,
        reduce_ptr_3 + num_blocks);

    *ke = thrust::reduce(reduce_ptr_4,
        reduce_ptr_4 + num_blocks);

    *press = thrust::reduce(reduce_ptr_5,
        reduce_ptr_5 + num_blocks);
}

