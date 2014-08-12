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

#include "cuda_common.hpp"
#include "kernel_files/generate_chunk_kernel.cuknl"

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
    cuda_chunk.generate_chunk_kernel(
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

    CUDALAUNCH(device_generate_chunk_kernel_init_cuda, density0, energy0, xvel0, yvel0, 
        state_density_d, state_energy_d, state_xvel_d, state_yvel_d);

    for (int state = 1; state < number_of_states; state++)
    {
        CUDALAUNCH(device_generate_chunk_kernel_cuda, 
            vertexx, vertexy, cellx, celly, density0, energy0, xvel0, yvel0,
            state_density_d, state_energy_d, state_xvel_d,
            state_yvel_d, state_xmin_d, state_xmax_d, state_ymin_d, state_ymax_d,
            state_radius_d, state_geometry_d, g_rect, g_circ, g_point, state);
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
}

