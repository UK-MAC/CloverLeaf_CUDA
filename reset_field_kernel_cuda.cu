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
 *  @brief CUDA reset field kernel.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details CUDA Copies all of the final end of step filed data to the begining of
 *  step data, ready for the next timestep.
 */

#include "cuda_common.hpp"
#include "kernel_files/reset_field_kernel.cuknl"

extern "C" void reset_field_kernel_cuda_
(void)
{
    cuda_chunk.reset_field_kernel();
}

void CloverleafCudaChunk::reset_field_kernel
(void)
{
    CUDALAUNCH(device_reset_field_kernel_cuda,
        density0, density1, energy0, energy1, xvel0, xvel1, yvel0, yvel1);
}

