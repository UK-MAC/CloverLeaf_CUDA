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

#include <iostream>
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_viscosity_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,
const double * __restrict const celldx,
const double * __restrict const celldy,
const double * __restrict const density0,
const double * __restrict const pressure,
      double * __restrict const viscosity,
const double * __restrict const xvel0,
const double * __restrict const yvel0)
{
    __kernel_indexes;

    double ugrad, vgrad, grad2, pgradx, pgrady, pgradx2, pgrady2,
        grad, ygrad, pgrad, xgrad, div, strain2, limiter;

    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        ugrad = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])
              - (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]);

        vgrad = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])
              - (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]);
        
        div = (celldx[column] * ugrad) + (celldy[row] * vgrad);

        strain2 = 0.5 * (xvel0[THARR2D(0, 1, 1)] + xvel0[THARR2D(1, 1, 1)]
                - xvel0[THARR2D(0, 0, 1)] - xvel0[THARR2D(1, 0, 1)])/celldy[row]
                + 0.5 * (yvel0[THARR2D(1, 0, 1)] + yvel0[THARR2D(1, 1, 1)]
                - yvel0[THARR2D(0, 0, 1)] - yvel0[THARR2D(0, 1, 1)])/celldx[column];

        pgradx = (pressure[THARR2D(1, 0, 0)] - pressure[THARR2D(-1, 0, 0)])
               / (celldx[column] + celldx[column + 1]);
        pgrady = (pressure[THARR2D(0, 1, 0)] - pressure[THARR2D(0, -1, 0)])
               / (celldy[row] + celldy[row + 1]);

        pgradx2 = pgradx*pgradx;
        pgrady2 = pgrady*pgrady;

        limiter = ((0.5 * ugrad / celldx[column]) * pgradx2
                + ((0.5 * vgrad / celldy[row]) * pgrady2)
                + (strain2 * pgradx * pgrady))
                / MAX(pgradx2 + pgrady2, 1.0e-16);

        if (limiter > 0 || div >= 0.0)
        {
            viscosity[THARR2D(0,0,0)] = 0.0;
        }
        else
        {
          pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);
          pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);
          pgrad = sqrt((pgradx * pgradx) + (pgrady * pgrady));

          xgrad = fabs(celldx[column] * pgrad / pgradx);
          ygrad = fabs(celldy[row] * pgrad / pgrady);

          grad = MIN(xgrad, ygrad);
          grad2 = grad * grad;

          viscosity[THARR2D(0,0,0)] = 2.0 * density0[THARR2D(0,0,0)] * grad2 * (limiter * limiter);
        }
    }
}

extern "C" void viscosity_kernel_cuda_
(void)
{
    chunk.viscosity_kernel();
}

void CloverleafCudaChunk::viscosity_kernel
(void)
{
    CUDA_BEGIN_PROFILE;

    device_viscosity_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, celldx, celldy, density0, pressure, viscosity,
        xvel0, yvel0);
    CUDA_ERR_CHECK;

    CUDA_END_PROFILE;
}

