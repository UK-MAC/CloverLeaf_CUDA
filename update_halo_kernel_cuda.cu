// TODO some copies are negative, or copy form slightly different locations - look through and fix
// change to use depth

#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"

#define CHUNK_left          0
#define CHUNK_right         1
#define CHUNK_bottom        2
#define CHUNK_top           3

#define EXTERNAL_FACE       (-1)

#define NUM_FIELDS          15

extern CudaDevPtrStorage pointer_storage;

extern CloverleafCudaChunk chunk;

/*
*   copies from the outer bit from [x_min, x_max] to the outer cells
*/

__global__ void device_update_halo_kernel_bottom_cuda
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_invert, int y_invert,
double* cur_array,
int depth)
{
    __large_kernel_indexes;
    // if its on the third row
    /*
    if(row == 2
    && column > 0 && column < x_max+3+x_extra)
    {
        for(int ii = 0; ii < depth; ii++)
        {
            //FIXME 1+k for all the ones that have x extra or y extra
            cur_array[THARR2D(0, -(1 + ii), x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, 0, x_extra)];
        }
    }
    */

    if(row == y_max)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, -1, x_extra)];
    }
    else if(row == y_max + 1)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, -3, x_extra)];
    }
    else if(row == y_max + 2)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, -5, x_extra)];
    }
}

__global__ void device_update_halo_kernel_top_cuda
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_invert, int y_invert,
double* cur_array,
int depth)
{
    __large_kernel_indexes;
    /*
    // if on the third row from the bottom
    if(row == y_max+1+y_extra
    && column > 0 && column < x_max+3+x_extra)
    {
        for(int ii = 0; ii < depth; ii++)
        {
            cur_array[THARR2D(0, 1 + ii, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, 0, x_extra)];
        }
    }
    */

    if(row == 3)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, 1, x_extra)];
    }
    else if(row == 2)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, 3, x_extra)];
    }
    else if(row == 1)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (y_invert?-1:1) * cur_array[THARR2D(0, 5, x_extra)];
    }
}

__global__ void device_update_halo_kernel_left_cuda
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_invert, int y_invert,
double* cur_array,
int depth)
{
    __large_kernel_indexes;
    /*
    // if on the third column from the left
    if(row > 0 && row < y_max+3+y_extra
    && column == 2)
    {
        for(int ii = 0; ii < depth; ii++)
        {
            cur_array[THARR2D(-(1 + ii), 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(0, 0, x_extra)];
        }
    }
    */

    if(column == 3)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(1, 0, x_extra)];
    }
    else if(column == 2)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(3, 0, x_extra)];
    }
    else if(column == 1)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(5, 0, x_extra)];
    }
}

__global__ void device_update_halo_kernel_right_cuda
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_invert, int y_invert,
double* cur_array,
int depth)
{
    __large_kernel_indexes;
    /*
    // if on the third column from the right
    if(row > 0 && row < y_max+3+y_extra
    && column == x_max+1+x_extra)
    {
        for(int ii = 0; ii < depth; ii++)
        {
            cur_array[THARR2D(1 + ii, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(0, 0, x_extra)];
        }
    }
    */

    if(column == x_max)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(-1, 0, x_extra)];
    }
    else if(column == x_max + 1)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(-3, 0, x_extra)];
    }
    else if(column == x_max + 2)
    {
        cur_array[THARR2D(0, 0, x_extra)] = (x_invert?-1:1) * cur_array[THARR2D(-5, 0, x_extra)];
    }
}

void update_array
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_invert, int y_invert,
const int* chunk_neighbours,
double* cur_array,
int depth)
{
    #define CHECK_LAUNCH(dir) \
    if(chunk_neighbours[CHUNK_ ## dir] == EXTERNAL_FACE)\
    {\
        device_update_halo_kernel_ ## dir ## _cuda<<< ((x_max+5)*(y_max+5))/BLOCK_SZ, BLOCK_SZ >>>\
        (x_min,x_max,y_min,y_max, x_extra, y_extra, x_invert, y_invert, cur_array, depth);\
        errChk(__LINE__, __FILE__);\
    }

    CHECK_LAUNCH(bottom);
    CHECK_LAUNCH(top);
    CHECK_LAUNCH(right);
    CHECK_LAUNCH(left);

}

void update_halo_cuda
(int x_min,int x_max,int y_min,int y_max,

const int* chunk_neighbours,

double* density0,
double* energy0,
double* pressure,
double* viscosity,
double* soundspeed,
double* density1,
double* energy1,
double* xvel0,
double* yvel0,
double* xvel1,
double* yvel1,
double* vol_flux_x,
double* vol_flux_y,
double* mass_flux_x,
double* mass_flux_y,

const int* fields,
int depth)
{

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif

    pointer_storage.setSize(x_max, y_max);
    double* cur_array_d ;

    #define HALO_UPDATE(arr, x_e, y_e, x_i, y_i) \
    {if(fields[FIELD_ ## arr] == 1)\
    {\
        cur_array_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, arr, BUFSZ2D(x_e, y_e));\
        update_array(x_min, x_max, y_min, y_max, \
            x_e, y_e, x_i, y_i, \
            chunk_neighbours, cur_array_d, depth);\
        pointer_storage.freeDevStorageAndCopy(cur_array_d, arr, BUFSZ2D(x_e, y_e));\
    }}

    HALO_UPDATE(density0, 0, 0, 0, 0);
    HALO_UPDATE(energy0, 0, 0, 0, 0);
    HALO_UPDATE(pressure, 0, 0, 0, 0);
    HALO_UPDATE(viscosity, 0, 0, 0, 0);
    HALO_UPDATE(soundspeed, 0, 0, 0, 0);
    HALO_UPDATE(density1, 0, 0, 0, 0);
    HALO_UPDATE(energy1, 0, 0, 0, 0);

    HALO_UPDATE(xvel0, 1, 1, 1, 0);
    HALO_UPDATE(yvel0, 1, 1, 0, 1);
    HALO_UPDATE(xvel1, 1, 1, 1, 0);
    HALO_UPDATE(yvel1, 1, 1, 0, 1);

    HALO_UPDATE(vol_flux_x, 1, 0, 1, 0);
    HALO_UPDATE(vol_flux_y, 0, 1, 0, 1);
    HALO_UPDATE(mass_flux_x, 1, 0, 1, 0);
    HALO_UPDATE(mass_flux_y, 0, 1, 0, 1);

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

extern "C" void update_halo_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max,
int* left,int* bottom,int* right,int* top,
int* left_boundary,int* bottom_boundary,int* right_boundary,int* top_boundary,

const int* chunk_neighbours,

double* density0,
double* energy0,
double* pressure,
double* viscosity,
double* soundspeed,
double* density1,
double* energy1,
double* xvel0,
double* yvel0,
double* xvel1,
double* yvel1,
double* vol_flux_x,
double* vol_flux_y,
double* mass_flux_x,
double* mass_flux_y,

const int* fields,
int* depth)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    update_halo_cuda(*x_min, *x_max, *y_min, *y_max,
        chunk_neighbours,
        density0, energy0, pressure, viscosity, soundspeed, density1, energy1,
        xvel0, yvel0, xvel1, yvel1,
        vol_flux_x, vol_flux_y, mass_flux_x, mass_flux_y,
        fields, *depth);
    #else
    chunk.update_halo_kernel(fields, *depth, chunk_neighbours);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::update_halo_kernel
(const int* fields, int depth,
const int* chunk_neighbours)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

    #define HALO_UPDATE_RESIDENT(arr, x_e, y_e, x_i, y_i) \
    {if(fields[FIELD_ ## arr] == 1)\
    {\
        update_array(x_min, x_max, y_min, y_max, \
            x_e, y_e, x_i, y_i, \
            chunk_neighbours, arr, depth);\
    }}

    HALO_UPDATE_RESIDENT(density0, 0, 0, 0, 0);
    HALO_UPDATE_RESIDENT(energy0, 0, 0, 0, 0);
    HALO_UPDATE_RESIDENT(pressure, 0, 0, 0, 0);
    HALO_UPDATE_RESIDENT(viscosity, 0, 0, 0, 0);
    HALO_UPDATE_RESIDENT(soundspeed, 0, 0, 0, 0);
    HALO_UPDATE_RESIDENT(density1, 0, 0, 0, 0);
    HALO_UPDATE_RESIDENT(energy1, 0, 0, 0, 0);

    HALO_UPDATE_RESIDENT(xvel0, 1, 1, 1, 0);
    HALO_UPDATE_RESIDENT(yvel0, 1, 1, 0, 1);
    HALO_UPDATE_RESIDENT(xvel1, 1, 1, 1, 0);
    HALO_UPDATE_RESIDENT(yvel1, 1, 1, 0, 1);

    HALO_UPDATE_RESIDENT(vol_flux_x, 1, 0, 1, 0);
    HALO_UPDATE_RESIDENT(vol_flux_y, 0, 1, 0, 1);
    HALO_UPDATE_RESIDENT(mass_flux_x, 1, 0, 1, 0);
    HALO_UPDATE_RESIDENT(mass_flux_y, 0, 1, 0, 1);

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}

