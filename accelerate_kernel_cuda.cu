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
 *  @brief CUDA acceleration kernel
 *  @author Michael Boulton NVIDIA Corporation
 *  @details The pressure and viscosity gradients are used to update the
 *  velocity field.
 */

#include "cuda_common.hpp"
#include "kernel_files/accelerate_kernel.cuknl"

extern "C" void accelerate_kernel_cuda_
(double *dbyt)
{
    cuda_chunk.accelerate_kernel(*dbyt);
}

void CloverleafCudaChunk::accelerate_kernel
(double dbyt)
{
    CUDALAUNCH(device_accelerate_kernel_cuda,
        dbyt, xarea, yarea, volume, density0,
        pressure, viscosity, xvel0, yvel0, xvel1, yvel1);
}

