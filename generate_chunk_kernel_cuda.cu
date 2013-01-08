
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "thrust/copy.h"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_generate_chunk_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* __restrict const vertexx,
const double* __restrict const vertexy,
const double* __restrict const cellx,
const double* __restrict const celly,
      double* __restrict const density0,
      double* __restrict const energy0,
      double* __restrict const xvel0,
      double* __restrict const yvel0,

const double* __restrict const state_density,
const double* __restrict const state_energy,
const double* __restrict const state_xvel,
const double* __restrict const state_yvel,
const double* __restrict const state_xmin,
const double* __restrict const state_xmax,
const double* __restrict const state_ymin,
const double* __restrict const state_ymax,
const double* __restrict const state_radius,
const int* __restrict const state_geometry,

const int g_rect,
const int g_circ,
const int state)
{
    __kernel_indexes;

    if(row < y_max+4
    && column < x_max+4)
    {
        if(state_geometry[state] == g_rect)
        {
            if(vertexx[column] >= state_xmin[state]
            && vertexx[column] <  state_xmax[state]
            && vertexy[row]    >= state_ymin[state]
            && vertexy[row]    <  state_ymax[state])
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density0[THARR2D(0, 0, 0)] = state_density[state];

                //unrolled do loop
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];
            }
        }
        else if(state_geometry[state] == g_circ)
        {
            double radius = sqrt(cellx[column] * cellx[column] + celly[row] + celly[row]);
            if(radius <= state_radius[state])
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density0[THARR2D(0, 0, 0)] = state_density[state];

                //unrolled do loop
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];
            }
        }
    }
}

__global__ void device_generate_chunk_kernel_init_cuda
(int x_min,int x_max,int y_min,int y_max,
      double* density0,
      double* energy0,
      double* xvel0,
      double* yvel0,
const double* state_density,
const double* state_energy,
const double* state_xvel,
const double* state_yvel)
{
    __kernel_indexes;

    if(row < y_max+4
    && column < x_max+4)
    {
        energy0[THARR2D(0, 0, 0)] = state_energy[0];
        density0[THARR2D(0, 0, 0)] = state_density[0];
        xvel0[THARR2D(0, 0, 1)] = state_xvel[0];
        yvel0[THARR2D(0, 0, 1)] = state_yvel[0];
    }
}

void generate_chunk_cuda
(int x_min,int x_max,int y_min,int y_max,
const double* vertexx,
const double* vertexy,
const double* cellx,
const double* celly,
      double* density0,
      double* energy0,
      double* xvel0,
      double* yvel0,

const int number_of_states,

const double* state_density,
const double* state_energy,
const double* state_xvel,
const double* state_yvel,
const double* state_xmin,
const double* state_xmax,
const double* state_ymin,
const double* state_ymax,
const double* state_radius,
const int* state_geometry,

const int g_rect,
const int g_circ)
{

    pointer_storage.setSize(x_max, y_max);

    double* density0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* energy0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* xvel0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* yvel0_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    double* vertexx_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vertexx, (x_max+5)*sizeof(double));
    double* vertexy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vertexy, (y_max+5)*sizeof(double));
    double* cellx_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, cellx, (x_max+4)*sizeof(double));
    double* celly_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celly, (y_max+4)*sizeof(double));

    double* state_density_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_density, number_of_states*sizeof(double));
    double* state_energy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_energy, number_of_states*sizeof(double));
    double* state_xvel_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_xvel, number_of_states*sizeof(double));
    double* state_yvel_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_yvel, number_of_states*sizeof(double));
    double* state_xmin_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_xmin, number_of_states*sizeof(double));
    double* state_xmax_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_xmax, number_of_states*sizeof(double));
    double* state_ymin_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_ymin, number_of_states*sizeof(double));
    double* state_ymax_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_ymax, number_of_states*sizeof(double));
    double* state_radius_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, state_radius, number_of_states*sizeof(double));

    //int* state_geometry_d = (int*)pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, (double*)state_geometry, number_of_states*sizeof(int));
    thrust::device_ptr<int> thr_geo = thrust::device_malloc<int>(number_of_states*sizeof(int));
    thrust::copy(state_geometry, state_geometry + number_of_states, thr_geo);
    int* state_geometry_d = thrust::raw_pointer_cast(thr_geo);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_generate_chunk_kernel_init_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, 
        density0_d, energy0_d, xvel0_d, yvel0_d, 
        state_density_d, state_energy_d, state_xvel_d, state_yvel_d);
    errChk(__LINE__, __FILE__);

    for(int state = 1; state < number_of_states; state++)
    {
        device_generate_chunk_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, 
            vertexx_d, vertexy_d, cellx_d, celly_d, density0_d, energy0_d, xvel0_d, yvel0_d, 
            state_density_d, state_energy_d, state_xvel_d,
            state_yvel_d, state_xmin_d, state_xmax_d, state_ymin_d, state_ymax_d,
            state_radius_d, state_geometry_d,  g_rect,  g_circ, state);
    }
    errChk(__LINE__, __FILE__);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    pointer_storage.freeDevStorageAndCopy(density0_d, density0, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(energy0_d, energy0, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(xvel0_d, xvel0, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(yvel0_d, yvel0, BUFSZ2D(0, 0));

    pointer_storage.freeDevStorage(vertexx_d);
    pointer_storage.freeDevStorage(vertexy_d);
    pointer_storage.freeDevStorage(cellx_d);
    pointer_storage.freeDevStorage(celly_d);
    pointer_storage.freeDevStorage(state_density_d);
    pointer_storage.freeDevStorage(state_energy_d);
    pointer_storage.freeDevStorage(state_xvel_d);
    pointer_storage.freeDevStorage(state_yvel_d);
    pointer_storage.freeDevStorage(state_xmin_d);
    pointer_storage.freeDevStorage(state_xmax_d);
    pointer_storage.freeDevStorage(state_ymin_d);
    pointer_storage.freeDevStorage(state_ymax_d);
    pointer_storage.freeDevStorage(state_radius_d);

    //pointer_storage.freeDevStorage((double*)state_geometry_d);
    thrust::device_free(thr_geo);

}

extern "C" void generate_chunk_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
const double* vertexx,
const double* vertexy,
const double* cellx,
const double* celly,
      double* density0,
      double* energy0,
      double* xvel0,
      double* yvel0,

const int* number_of_states,

const double* state_density,
const double* state_energy,
const double* state_xvel,
const double* state_yvel,
const double* state_xmin,
const double* state_xmax,
const double* state_ymin,
const double* state_ymax,
const double* state_radius,
const int* state_geometry,

const int* g_rect,
const int* g_circ)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    generate_chunk_cuda(*x_min, *x_max, *y_min, *y_max, 
        vertexx, vertexy, cellx, celly, density0, energy0, xvel0, yvel0, 
        * number_of_states, state_density, state_energy, state_xvel,
        state_yvel, state_xmin, state_xmax, state_ymin, state_ymax,
        state_radius, state_geometry, * g_rect, * g_circ);
    #else
    chunk.generate_chunk_kernel(
        * number_of_states, state_density, state_energy, state_xvel,
        state_yvel, state_xmin, state_xmax, state_ymin, state_ymax,
        state_radius, state_geometry, * g_rect, * g_circ);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::generate_chunk_kernel
(const int number_of_states, 
const double* state_density, const double* state_energy,
const double* state_xvel, const double* state_yvel,
const double* state_xmin, const double* state_xmax,
const double* state_ymin, const double* state_ymax,
const double* state_radius, const int* state_geometry,
const int g_rect, const int g_circ)
{

    //only copied and used one time, don't care much about speed...
    #define THRUST_ALLOC_ARRAY(arr) \
        thrust::device_ptr<double> thr_state_ ## arr ## _d  \
        = thrust::device_malloc<double>(number_of_states*sizeof(double));\
        thrust::copy(state_ ## arr , state_ ## arr  + number_of_states, thr_state_ ## arr ## _d);\
        const double* state_ ## arr ## _d = thrust::raw_pointer_cast(thr_state_ ## arr ## _d);
    THRUST_ALLOC_ARRAY(density);
    THRUST_ALLOC_ARRAY(energy);
    THRUST_ALLOC_ARRAY(xvel);
    THRUST_ALLOC_ARRAY(yvel);
    THRUST_ALLOC_ARRAY(xmin);
    THRUST_ALLOC_ARRAY(xmax);
    THRUST_ALLOC_ARRAY(ymin);
    THRUST_ALLOC_ARRAY(ymax);
    THRUST_ALLOC_ARRAY(radius);

    thrust::device_ptr<int> thr_state_geometry_d = thrust::device_malloc<int>(number_of_states*sizeof(int));
    thrust::copy(state_geometry, state_geometry + number_of_states, thr_state_geometry_d);
    const int* state_geometry_d = thrust::raw_pointer_cast(thr_state_geometry_d);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_generate_chunk_kernel_init_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, 
        density0, energy0, xvel0, yvel0, 
        state_density_d, state_energy_d, state_xvel_d, state_yvel_d);
    errChk(__LINE__, __FILE__);

    for(int state = 1; state < number_of_states; state++)
    {
        device_generate_chunk_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, 
            vertexx, vertexy, cellx, celly, density0, energy0, xvel0, yvel0, 
            state_density_d, state_energy_d, state_xvel_d,
            state_yvel_d, state_xmin_d, state_xmax_d, state_ymin_d, state_ymax_d,
            state_radius_d, state_geometry_d,  g_rect,  g_circ, state);
        errChk(__LINE__, __FILE__);
    }
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    thrust::device_free(thr_state_density_d);
    thrust::device_free(thr_state_energy_d);
    thrust::device_free(thr_state_xvel_d);
    thrust::device_free(thr_state_yvel_d);
    thrust::device_free(thr_state_xmin_d);
    thrust::device_free(thr_state_xmax_d);
    thrust::device_free(thr_state_radius_d);
    thrust::device_free(thr_state_geometry_d);

}
