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
 *  @brief CUDA field summary kernel
 *  @author Michael Boulton NVIDIA Corporation
 *  @details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include "cuda_common.hpp"
#include "kernel_files/field_summary_kernel.cuknl"
#include "host_reductions_kernel_cuda.hpp"

extern "C" void field_summary_kernel_cuda_
(double* vol, double* mass, double* ie, double* ke, double* press)
{
    cuda_chunk.field_summary_kernel(vol, mass, ie, ke, press);
}

void CloverleafCudaChunk::field_summary_kernel
(double* vol, double* mass, double* ie, double* ke, double* press)
{
    CUDALAUNCH(device_field_summary_kernel_cuda, volume, density0,
        energy0, pressure, xvel0, yvel0,
        reduce_buf_1, reduce_buf_2, reduce_buf_3,
        reduce_buf_4, reduce_buf_5);

    ReduceToHost<double>::sum(reduce_buf_1, vol, num_blocks);
    ReduceToHost<double>::sum(reduce_buf_2, mass, num_blocks);
    ReduceToHost<double>::sum(reduce_buf_3, ie, num_blocks);
    ReduceToHost<double>::sum(reduce_buf_4, ke, num_blocks);
    ReduceToHost<double>::sum(reduce_buf_5, press, num_blocks);
}

