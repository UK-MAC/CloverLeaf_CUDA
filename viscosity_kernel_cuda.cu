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
 *  @brief CUDA viscosity kernel.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Calculates an artificial viscosity using the Wilkin's method to
 *  smooth out shock front and prevent oscillations around discontinuities.
 *  Only cells in compression will have a non-zero value.
 */

#include "cuda_common.hpp"
#include "kernel_files/viscosity_kernel.cuknl"

extern "C" void viscosity_kernel_cuda_
(void)
{
    cuda_chunk.viscosity_kernel();
}

void CloverleafCudaChunk::viscosity_kernel
(void)
{
    CUDALAUNCH(device_viscosity_kernel_cuda,
        celldx, celldy, density0, pressure, viscosity,
        xvel0, yvel0);
}

