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
 *  @brief CUDA flux kernel
 *  @author Michael Boulton NVIDIA Corporation
 *  @details The edge volume fluxes are calculated based on the velocity fields.
 */

#include "cuda_common.hpp"
#include "kernel_files/flux_calc_kernel.cuknl"

extern "C" void flux_calc_kernel_cuda_
(double *dbyt)
{
    cuda_chunk.flux_calc_kernel(*dbyt);
}

void CloverleafCudaChunk::flux_calc_kernel
(double dbyt)
{
    CUDALAUNCH(device_flux_calc_kernel_cuda,
        dbyt, xarea, yarea, xvel0, yvel0,
        xvel1, yvel1, vol_flux_x, vol_flux_y);
}

