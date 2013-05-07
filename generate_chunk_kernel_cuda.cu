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
 *  @brief CUDA mesh chunk generation driver
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Invoked the users specified chunk generator.
 */

#include <iostream>
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "thrust/copy.h"

#include "chunk_cuda.cu"

extern CloverleafCudaChunk chunk;

__global__ void device_generate_chunk_kernel_init_cuda
(int x_min, int x_max, int y_min, int y_max,
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

    if (row >= (y_min + 1) - 2 && row <= (y_max + 1) + 2
    && column >= (x_min + 1) - 2 && column <= (x_max + 1) + 2)
    {
        energy0[THARR2D(0, 0, 0)] = state_energy[0];
        density0[THARR2D(0, 0, 0)] = state_density[0];
        xvel0[THARR2D(0, 0, 1)] = state_xvel[0];
        yvel0[THARR2D(0, 0, 1)] = state_yvel[0];
    }
}

__global__ void device_generate_chunk_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,
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
const int g_point,
const int state)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - 2 && row <= (y_max + 1) + 2
    && column >= (x_min + 1) - 2 && column <= (x_max + 1) + 2)
    {
        if (state_geometry[state] == g_rect)
        {
            if (vertexx[column+1] >= state_xmin[state]
            && vertexx[column] <  state_xmax[state]
            && vertexy[row+1]    >= state_ymin[state]
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
        else if (state_geometry[state] == g_circ)
        {
            double radius = sqrt(cellx[column] * cellx[column] + celly[row] + celly[row]);
            if (radius <= state_radius[state])
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
        else if (state_geometry[state] == g_point)
        {
            if (vertexx[column] == state_xmin[state]
            && vertexy[row] == state_ymin[state])
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

extern "C" void generate_chunk_kernel_cuda_
(const int* number_of_states,
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
const int* g_circ,
const int* g_point)
{
    chunk.generate_chunk_kernel(
        *number_of_states, state_density, state_energy, state_xvel,
        state_yvel, state_xmin, state_xmax, state_ymin, state_ymax,
        state_radius, state_geometry, *g_rect, *g_circ, *g_point);
}

void CloverleafCudaChunk::generate_chunk_kernel
(const int number_of_states, 
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
const int g_circ,
const int g_point)
{
    // only copied and used one time, don't care about speed.
    #define THRUST_ALLOC_ARRAY(arr, type) \
        thrust::device_ptr<type> thr_state_ ## arr ## _d  \
            = thrust::device_malloc<type>(number_of_states*sizeof(type));\
        thrust::copy(state_ ## arr , \
                     state_ ## arr  + number_of_states, \
                     thr_state_ ## arr ## _d);\
        const type* state_ ## arr ## _d  \
            = thrust::raw_pointer_cast(thr_state_ ## arr ## _d);

    CUDA_BEGIN_PROFILE;

    THRUST_ALLOC_ARRAY(density, double);
    THRUST_ALLOC_ARRAY(energy, double);
    THRUST_ALLOC_ARRAY(xvel, double);
    THRUST_ALLOC_ARRAY(yvel, double);
    THRUST_ALLOC_ARRAY(xmin, double);
    THRUST_ALLOC_ARRAY(xmax, double);
    THRUST_ALLOC_ARRAY(ymin, double);
    THRUST_ALLOC_ARRAY(ymax, double);
    THRUST_ALLOC_ARRAY(radius, double);
    THRUST_ALLOC_ARRAY(geometry, int);

    #undef THRUST_ALLOC_ARRAY

    device_generate_chunk_kernel_init_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, density0, energy0, xvel0, yvel0, 
        state_density_d, state_energy_d, state_xvel_d, state_yvel_d);
    CUDA_ERR_CHECK;

    for (int state = 1; state < number_of_states; state++)
    {
        device_generate_chunk_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, 
            vertexx, vertexy, cellx, celly, density0, energy0, xvel0, yvel0, 
            state_density_d, state_energy_d, state_xvel_d,
            state_yvel_d, state_xmin_d, state_xmax_d, state_ymin_d, state_ymax_d,
            state_radius_d, state_geometry_d, g_rect, g_circ, g_point, state);
        CUDA_ERR_CHECK;
    }

    thrust::device_free(thr_state_density_d);
    thrust::device_free(thr_state_energy_d);
    thrust::device_free(thr_state_xvel_d);
    thrust::device_free(thr_state_yvel_d);
    thrust::device_free(thr_state_xmin_d);
    thrust::device_free(thr_state_xmax_d);
    thrust::device_free(thr_state_ymin_d);
    thrust::device_free(thr_state_ymax_d);
    thrust::device_free(thr_state_radius_d);
    thrust::device_free(thr_state_geometry_d);

    CUDA_END_PROFILE;
}

