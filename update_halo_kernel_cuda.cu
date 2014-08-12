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
 *  @brief CUDA kernel to update the external halo cells in a chunk.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
 */

#include "cuda_common.hpp"
#include "kernel_files/update_halo_kernel.cuknl"

extern "C" void update_halo_kernel_cuda_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    cuda_chunk.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void CloverleafCudaChunk::update_array
(int x_min, int x_max, int y_min, int y_max,
cell_info_t const& grid_type,
const int* chunk_neighbours,
double* cur_array_d,
int depth)
{
    #define CHECK_LAUNCH(face, dir)                                     \
    if (EXTERNAL_FACE == chunk_neighbours[CHUNK_ ## face - 1])          \
    {                                                                   \
        if (profiler_on)                                                \
        {                                                               \
            cudaEventCreate(&_t0);                                      \
            cudaEventRecord(_t0);                                       \
        }                                                               \
        const int launch_sz = (ceil((dir##_max+5+grid_type.dir##_extra) \
            /static_cast<float>(BLOCK_SZ))) * depth;                    \
        device_update_halo_kernel_##face##_cuda                         \
        <<<launch_sz, BLOCK_SZ >>>                                      \
            (x_min, x_max, y_min, y_max, grid_type, cur_array_d, depth);\
        CUDA_ERR_CHECK;                                                 \
        if (profiler_on)                                                \
        {                                                               \
            cudaEventCreate(&_t1);                                      \
            cudaEventRecord(_t1);                                       \
            cudaEventSynchronize(_t1);                                  \
            cudaEventElapsedTime(&taken, _t0, _t1);                     \
            std::string func_name("device_update_halo_kernel_"#face);   \
            if (kernel_times.end() != kernel_times.find(func_name))     \
            {                                                           \
                kernel_times.at(func_name) += taken;                    \
            }                                                           \
            else                                                        \
            {                                                           \
                kernel_times[func_name] = taken;                        \
            }                                                           \
        }                                                               \
    }
    CHECK_LAUNCH(bottom, x);
    CHECK_LAUNCH(top, x);
    CHECK_LAUNCH(left, y);
    CHECK_LAUNCH(right, y);

    #undef CHECK_LAUNCH
}

void CloverleafCudaChunk::update_halo_kernel
(const int* fields,
const int depth,
const int* chunk_neighbours)
{
    #define HALO_UPDATE_RESIDENT(arr, grid_type)        \
    {if (1 == fields[FIELD_##arr - 1])                  \
    {                                                   \
        update_array(x_min, x_max, y_min, y_max,        \
            grid_type, chunk_neighbours, arr, depth);   \
    }}

    HALO_UPDATE_RESIDENT(density0, CELL);
    HALO_UPDATE_RESIDENT(density1, CELL);
    HALO_UPDATE_RESIDENT(energy0, CELL);
    HALO_UPDATE_RESIDENT(energy1, CELL);
    HALO_UPDATE_RESIDENT(pressure, CELL);
    HALO_UPDATE_RESIDENT(viscosity, CELL);

    HALO_UPDATE_RESIDENT(xvel0, VERTEX_X);
    HALO_UPDATE_RESIDENT(xvel1, VERTEX_X);

    HALO_UPDATE_RESIDENT(yvel0, VERTEX_Y);
    HALO_UPDATE_RESIDENT(yvel1, VERTEX_Y);

    HALO_UPDATE_RESIDENT(vol_flux_x, X_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_x, X_FACE);

    HALO_UPDATE_RESIDENT(vol_flux_y, Y_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_y, Y_FACE);

    #undef HALO_UPDATE_RESIDENT
}

