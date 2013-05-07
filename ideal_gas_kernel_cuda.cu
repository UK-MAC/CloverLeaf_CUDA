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
 *  @brief CUDA ideal gas kernel.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Calculates the pressure and sound speed for the mesh chunk using
 *  the ideal gas equation of state, with a fixed gamma of 1.4.
 */


#include <iostream>
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_ideal_gas_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,
const double * __restrict const density,
const double * __restrict const energy,
      double * __restrict const pressure,
      double * __restrict const soundspeed)
{
    __kernel_indexes;
    double v, pressurebyenergy, pressurebyvolume, sound_speed_squared;

    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        v = 1.0 / density[THARR2D(0, 0, 0)];

        pressure[THARR2D(0, 0, 0)] = (1.4 - 1.0) * density[THARR2D(0, 0, 0)] * energy[THARR2D(0, 0, 0)];

        pressurebyenergy = (1.4 - 1.0) * density[THARR2D(0, 0, 0)];

        pressurebyvolume = - density[THARR2D(0, 0, 0)] * pressure[THARR2D(0, 0, 0)];

        sound_speed_squared = v * v 
            * (pressure[THARR2D(0, 0, 0)] * pressurebyenergy - pressurebyvolume);

        soundspeed[THARR2D(0, 0, 0)] = sqrt(sound_speed_squared);
    }
}

extern "C" void ideal_gas_kernel_cuda_predict_
(void)
{
    chunk.ideal_gas_kernel(1);
}

extern "C" void ideal_gas_kernel_cuda_nopredict_
(void)
{
    chunk.ideal_gas_kernel(0);
}

void CloverleafCudaChunk::ideal_gas_kernel
(int predict)
{
    CUDA_BEGIN_PROFILE;

    if (predict)
    {
        device_ideal_gas_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min,x_max,y_min,y_max, density1, energy1, pressure, soundspeed);
    }
    else
    {
        device_ideal_gas_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min,x_max,y_min,y_max, density0, energy0, pressure, soundspeed);
    }
    CUDA_ERR_CHECK;

    CUDA_END_PROFILE;
}

