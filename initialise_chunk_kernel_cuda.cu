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

#include "cuda_common.hpp"
#include "kernel_files/initialise_chunk_kernel.cuknl"

extern "C" void initialise_chunk_kernel_cuda_
(double* d_xmin, double* d_ymin, double* d_dx, double* d_dy)
{
    cuda_chunk.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
}

void CloverleafCudaChunk::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    CUDALAUNCH(device_initialise_chunk_kernel_vertex_cuda,
        d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy);

    CUDALAUNCH(device_initialise_chunk_kernel_cuda,
        d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy,
        cellx, celldx, celly, celldy,
        volume, xarea, yarea);
}

