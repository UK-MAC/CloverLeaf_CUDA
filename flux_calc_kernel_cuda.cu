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

#include <iostream>
#include "cuda_common.cu"
#include "ftocmacros.h"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_flux_calc_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
double dt,
const double * __restrict const xarea,
const double * __restrict const yarea,
const double * __restrict const xvel0,
const double * __restrict const yvel0,
const double * __restrict const xvel1,
const double * __restrict const yvel1,
      double * __restrict const vol_flux_x,
      double * __restrict const vol_flux_y)
{
    __kernel_indexes;

    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1) + 1)
    {
        vol_flux_x[THARR2D(0, 0, 1)] = 0.25 * dt * xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]);
    }

    if (row >= (y_min + 1) && row <= (y_max + 1) + 1
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        vol_flux_y[THARR2D(0, 0, 0)] = 0.25 * dt * yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]);
    }

}

extern "C" void flux_calc_kernel_cuda_
(double *dbyt)
{
    chunk.flux_calc_kernel(*dbyt);
}

void CloverleafCudaChunk::flux_calc_kernel
(double dbyt)
{
    CUDA_BEGIN_PROFILE;

    device_flux_calc_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, dbyt, xarea, yarea, xvel0, yvel0,
        xvel1, yvel1, vol_flux_x, vol_flux_y);
    CUDA_ERR_CHECK;

    CUDA_END_PROFILE;
}

