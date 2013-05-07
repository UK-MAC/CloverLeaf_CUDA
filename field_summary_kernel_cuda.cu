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
 *  @brief CUDA field summary kernel
 *  @author Michael Boulton NVIDIA Corporation
 *  @details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include <iostream>
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_field_summary_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,
const double* __restrict const volume,
const double* __restrict const density0,
const double* __restrict const energy0,
const double* __restrict const pressure,
const double* __restrict const xvel0,
const double* __restrict const yvel0,
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

    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        double vsqrd = 0.0;

        // unrolled do loop
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
    for (int offset = BLOCK_SZ / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            vol_shared[threadIdx.x] += vol_shared[threadIdx.x + offset];
            mass_shared[threadIdx.x] += mass_shared[threadIdx.x + offset];
            ie_shared[threadIdx.x] += ie_shared[threadIdx.x + offset];
            ke_shared[threadIdx.x] += ke_shared[threadIdx.x + offset];
            press_shared[threadIdx.x] += press_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    vol[blockIdx.x] = vol_shared[0];
    mass[blockIdx.x] = mass_shared[0];
    ie[blockIdx.x] = ie_shared[0];
    ke[blockIdx.x] = ke_shared[0];
    press[blockIdx.x] = press_shared[0];

}

extern "C" void field_summary_kernel_cuda_
(double* vol, double* mass, double* ie, double* ke, double* press)
{
    chunk.field_summary_kernel(vol, mass, ie, ke, press);
}

void CloverleafCudaChunk::field_summary_kernel
(double* vol, double* mass, double* ie, double* ke, double* press)
{
    CUDA_BEGIN_PROFILE;

    device_field_summary_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, volume, density0,
        energy0, pressure, xvel0, yvel0,
        work_array_1, work_array_2, work_array_3,
        work_array_4, work_array_5);
    CUDA_ERR_CHECK;

    *vol = thrust::reduce(reduce_ptr_1,
                          reduce_ptr_1 + num_blocks,
                          0.0);

    *mass = thrust::reduce(reduce_ptr_2,
                           reduce_ptr_2 + num_blocks,
                           0.0);

    *ie = thrust::reduce(reduce_ptr_3,
                         reduce_ptr_3 + num_blocks,
                         0.0);

    *ke = thrust::reduce(reduce_ptr_4,
                         reduce_ptr_4 + num_blocks,
                         0.0);

    *press = thrust::reduce(reduce_ptr_5,
                            reduce_ptr_5 + num_blocks,
                            0.0);

    CUDA_END_PROFILE;
}

