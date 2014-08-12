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

#include "cuda_common.hpp"
#include "kernel_files/ideal_gas_kernel.cuknl"

extern "C" void ideal_gas_kernel_predict_cuda_
(void)
{
    cuda_chunk.ideal_gas_kernel(1);
}

extern "C" void ideal_gas_kernel_nopredict_cuda_
(void)
{
    cuda_chunk.ideal_gas_kernel(0);
}

void CloverleafCudaChunk::ideal_gas_kernel
(int predict)
{
    if (predict)
    {
        CUDALAUNCH(device_ideal_gas_kernel_cuda, density1, energy1, pressure, soundspeed);
    }
    else
    {
        CUDALAUNCH(device_ideal_gas_kernel_cuda, density0, energy0, pressure, soundspeed);
    }
}

