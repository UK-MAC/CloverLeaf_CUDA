/*Crown Copyright 2012 AWE.
 *
 * This file is part of CloverLeaf.
 *
 * CloverLeaf is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * CloverLeaf is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

/*
 *  @brief CUDA common file
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Contains common elements for cuda kernels
 */

#ifndef __CUDA_COMMON_INC
#define __CUDA_COMMON_INC

// size of workgroup/block
#define BLOCK_SZ 256

// number of bytes to allocate for x size array
#define BUFSZX(x_extra)   \
    ( ((x_max) + 4 + x_extra)       \
    * sizeof(double) )

// number of bytes to allocate for y size array
#define BUFSZY(y_extra)   \
    ( ((y_max) + 4 + y_extra)       \
    * sizeof(double) )

// number of bytes to allocate for 2d array
#define BUFSZ2D(x_extra, y_extra)   \
    ( ((x_max) + 4 + x_extra)       \
    * ((y_max) + 4 + y_extra)       \
    * sizeof(double) )

// access a volue in a 2d array given the x and y offset from current thread
// index, adding or subtracting a bit more if it is one of the arrays with
// bigger rows
#define THARR2D(x_offset, y_offset, big_row)\
    ( glob_id                               \
    + (x_offset)                            \
    + ((y_offset) * (x_max + 4))            \
    + (big_row * (row + (y_offset))) )

// kernel indexes uses in all kernels
#define __kernel_indexes                    \
    const int glob_id = threadIdx.x         \
        + blockIdx.x * blockDim.x;          \
    const int row = glob_id / (x_max + 4);  \
    const int column = glob_id % (x_max + 4);

// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density0      0
#define FIELD_density1      1
#define FIELD_energy0       2
#define FIELD_energy1       3
#define FIELD_pressure      4
#define FIELD_viscosity     5
#define FIELD_soundspeed    6
#define FIELD_xvel0         7
#define FIELD_xvel1         8
#define FIELD_yvel0         9
#define FIELD_yvel1         10
#define FIELD_vol_flux_x    11
#define FIELD_vol_flux_y    12
#define FIELD_mass_flux_x   13
#define FIELD_mass_flux_y   14
#define NUM_FIELDS          15

enum {CELL_DATA, VERTEX_DATA, X_FACE_DATA, Y_FACE_DATA};

/*******************/

#include <iostream>
#include <vector>
#include <algorithm>
#include "omp.h"
#include "ftocmacros.h"

// for reduction in calc_dt and PdV
#include "thrust/copy.h"
#include "thrust/reduce.h"
#include "thrust/fill.h"
#include "thrust/functional.h"
#include "thrust/device_allocator.h"
#include "thrust/device_malloc.h"

/*******************/

// lots of time is spent error checking - define this to stop checking for errors.
#ifndef NO_ERR_CHK

#ifdef _GNUC_
void errChk(int, std::string const&) __attribute__((always_inline));
#endif

inline void errChk
(int line_num, std::string const& file)
{
    cudaDeviceSynchronize();
    int l_e = cudaGetLastError();
    if(l_e != cudaSuccess)
    {
        std::cout << "error on line " << line_num << " of ";
        std::cout << file << std::endl;
        std::cout << "return code " << l_e; 
        std::cout << std::endl;
        exit(l_e);
    }
}

#else

// do nothing instead
#define errChk(l, f) ;

#endif //NO_ERR_CHK

// whether to time kernel run times
#ifdef TIME_KERNELS

// beginning of profiling bit
#define _CUDA_BEGIN_PROFILE_name(x) \
    double x##t_0, x##t_1;          \
    x##t_0 = omp_get_wtime();

// end of profiling bit
#define _CUDA_END_PROFILE_name(x)                   \
    cudaDeviceSynchronize();                        \
    x##t_1 = omp_get_wtime();                       \
    std::cout << "[PROFILING] " << x##t_1 - x##t_0  \
    << " to calculate block \"" << #x <<            \
    "\" in " << __FILE__  <<std::endl;

#else

#define _CUDA_BEGIN_PROFILE_name(x) ;
#define _CUDA_END_PROFILE_name(x) ;

#endif // TIME_KERNELS

typedef struct cell_info {
    const int x_e;
    const int y_e;
    const int x_i;
    const int y_i;
    const int x_f;
    const int y_f;
    const int grid_type;

    cell_info
    (int x_extra, int y_extra,
    int x_invert, int y_invert,
    int x_face, int y_face,
    int in_type)
    :x_e(x_extra), y_e(y_extra),
    x_i(x_invert), y_i(y_invert),
    x_f(x_face), y_f(y_face),
    grid_type(in_type)
    {
        ;
    }

} cell_info_t;

// types of array data
const static cell_info_t CELL(    0, 0,  1,  1, 0, 0, CELL_DATA);
const static cell_info_t VERTEX_X(1, 1, -1,  1, 0, 0, VERTEX_DATA);
const static cell_info_t VERTEX_Y(1, 1,  1, -1, 0, 0, VERTEX_DATA);
const static cell_info_t X_FACE(  1, 0, -1,  1, 1, 0, X_FACE_DATA);
const static cell_info_t Y_FACE(  0, 1,  1, -1, 0, 1, Y_FACE_DATA);

__device__ inline static int sum_func (int x, int y) { return x + y; }
__device__ inline static int min_func (int x, int y) { return MIN(x, y); }
__device__ inline static int max_func (int x, int y) { return MAX(x, y); }

__device__ inline static double sum_func (double x, double y) { return x + y; }
__device__ inline static double min_func (double x, double y) { return MIN(x, y); }
__device__ inline static double max_func (double x, double y) { return MAX(x, y); }

template < int offset >
class Reduce
{
public:
    __device__ inline static void run
    (double* array, double* out, double(*func)(double, double))
    {
        // only need to synch if not working within a warp
        if (offset > 16)
        {
            __syncthreads();
        }

        // only continue if it's in the lower half
        if (threadIdx.x < offset)
        {
            array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
            Reduce< offset/2 >::run(array, out, func);
        }
    }

    __device__ inline static void run
    (int* array, int* out, int(*func)(int, int))
    {
        // only need to synch if not working within a warp
        if (offset > 16)
        {
            __syncthreads();
        }

        // only continue if it's in the lower half
        if (threadIdx.x < offset)
        {
            array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
            Reduce< offset/2 >::run(array, out, func);
        }
    }
};

template < >
class Reduce < 0 >
{
public:
    __device__ inline static void run
    (double* array, double* out, double(*func)(double, double))
    {
        out[blockIdx.x] = array[0];
    }

    __device__ inline static void run
    (int* array, int* out, int(*func)(int, int))
    {
        out[blockIdx.x] = array[0];
    }
};

#endif

