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
 *  @brief CUDA driver for chunk initialisation.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Invokes the user specified chunk initialisation kernel.
 */


#include <iostream>
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_initialise_chunk_kernel_vertex_cuda
(int x_min, int x_max, int y_min, int y_max,
const double d_xmin,
const double d_ymin,
const double d_dx,
const double d_dy,
double* __restrict const vertexx,
double* __restrict const vertexdx,
double* __restrict const vertexy,
double* __restrict const vertexdy)
{
    const int glob_id = threadIdx.x
        + blockIdx.x * blockDim.x;

    //bigger indexes 
    const int row = glob_id / (x_max + 5);
    const int column = glob_id % (x_max + 5);

    //fill out x arrays
    if (row == 0)
    {
        vertexx[column] = d_xmin + d_dx
            * static_cast<double>((column - 1) - x_min);
        vertexdx[column] = d_dx;
    }

    // fill out y arrays
    if (column == 0)
    {
        vertexy[row] = d_ymin + d_dy
            * static_cast<double>((row - 1) - y_min);
        vertexdy[row] = d_dy;
    }

}

__global__ void device_initialise_chunk_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,
const double d_xmin,
const double d_ymin,
const double d_dx,
const double d_dy,
const double* __restrict const vertexx,
const double* __restrict const vertexdx,
const double* __restrict const vertexy,
const double* __restrict const vertexdy,
      double* __restrict const cellx,
      double* __restrict const celldx,
      double* __restrict const celly,
      double* __restrict const celldy,
      double* __restrict const volume, 
      double* __restrict const xarea, 
      double* __restrict const yarea)
{
    __kernel_indexes;

    //fill x arrays
    if (row == 0)
    {
        cellx[column] = 0.5 * (vertexx[column] + vertexx[column + 1]);
        celldx[column] = d_dx;
    }

    //fill y arrays
    if (column == 0)
    {
        celly[row] = 0.5 * (vertexy[row] + vertexy[row + 1]);
        celldy[row] = d_dy;
    }

    if (row >= (y_min + 1) - 2 && row <= (y_max + 1) + 2
    && column >= (x_min + 1) - 2 && column <= (x_max + 1) + 2)
    {
        volume[THARR2D(0, 0, 0)] = d_dx * d_dy;
        xarea[THARR2D(0, 0, 1)] = d_dy;
        yarea[THARR2D(0, 0, 0)] = d_dx;
    }

}

extern "C" void initialise_chunk_kernel_cuda_
(double* d_xmin, double* d_ymin, double* d_dx, double* d_dy)
{
    chunk.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
}

void CloverleafCudaChunk::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    CUDA_BEGIN_PROFILE;

    device_initialise_chunk_kernel_vertex_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy);
    CUDA_ERR_CHECK;

    device_initialise_chunk_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy,
        cellx, celldx, celly, celldy,
        volume, xarea, yarea);
    CUDA_ERR_CHECK;

    CUDA_END_PROFILE;
}

