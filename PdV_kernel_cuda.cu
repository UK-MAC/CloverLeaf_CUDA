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
 *  @brief CUDA PdV kernel.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Calculates the change in energy and density in a cell using the
 *  change on cell volume due to the velocity gradients in a cell. The time
 *  level of the velocity data depends on whether it is invoked as the
 *  predictor or corrector.
 */

#include <iostream>
#include "cuda_common.cu"
#include "ftocmacros.h"

#include <algorithm>

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_PdV_cuda_kernel_predict
(int x_min, int x_max, int y_min, int y_max, double dt,
         int * __restrict const error_condition,
const double * __restrict const xarea,
const double * __restrict const yarea,
const double * __restrict const volume,
const double * __restrict const density0,
      double * __restrict const density1,
const double * __restrict const energy0,
      double * __restrict const energy1,
const double * __restrict const pressure,
const double * __restrict const viscosity,
const double * __restrict const xvel0,
const double * __restrict const yvel0,
const double * __restrict const xvel1,
const double * __restrict const yvel1)
{
    __kernel_indexes;

    __shared__ int err_condition_shared[BLOCK_SZ];
    err_condition_shared[threadIdx.x] = 0;

    double volume_change;
    double recip_volume, energy_change, min_cell_volume,
        right_flux, left_flux, top_flux, bottom_flux, total_flux;
    
    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        left_flux   = (xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)] 
            + xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]))
            * 0.25 * dt * 0.5;
        right_flux  = (xarea[THARR2D(1, 0, 1)]
            * (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)] 
            + xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)]))
            * 0.25 * dt * 0.5;

        bottom_flux = (yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)] 
            + yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]))
            * 0.25 * dt * 0.5;
        top_flux    = (yarea[THARR2D(0, 1, 0)]
            * (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)] 
            + yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)]))
            * 0.25 * dt * 0.5;

        total_flux = right_flux - left_flux + top_flux - bottom_flux;

        volume_change = volume[THARR2D(0, 0, 0)]
            / (volume[THARR2D(0, 0, 0)] + total_flux);

        //minimum of total, horizontal, and vertical flux
        min_cell_volume = 
            MIN(volume[THARR2D(0, 0, 0)] + total_flux,
            MIN(volume[THARR2D(0, 0, 0)] + right_flux - left_flux,
                volume[THARR2D(0, 0, 0)] + top_flux - bottom_flux));

        if (volume_change <= 0.0)
        {
            err_condition_shared[threadIdx.x] = 1;
        }
        if (min_cell_volume <= 0.0)
        {
            err_condition_shared[threadIdx.x] = 2;
        }

        recip_volume = 1.0/volume[THARR2D(0, 0, 0)];

        energy_change = ((pressure[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)])
            + (viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)]))
            * total_flux * recip_volume;

        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)] - energy_change;
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)] * volume_change;
    }

    Reduce< BLOCK_SZ/2 >::run(err_condition_shared, error_condition, max_func);
}

__global__ void device_PdV_cuda_kernel_not_predict
(int x_min, int x_max, int y_min, int y_max, double dt,
         int * __restrict const error_condition,
const double * __restrict const xarea,
const double * __restrict const yarea,
const double * __restrict const volume,
const double * __restrict const density0,
      double * __restrict const density1,
const double * __restrict const energy0,
      double * __restrict const energy1,
const double * __restrict const pressure,
const double * __restrict const viscosity,
const double * __restrict const xvel0,
const double * __restrict const yvel0,
const double * __restrict const xvel1,
const double * __restrict const yvel1)
{
    __kernel_indexes;

    __shared__ int err_condition_shared[BLOCK_SZ];
    err_condition_shared[threadIdx.x] = 0;

    double volume_change;
    double recip_volume, energy_change, min_cell_volume,
        right_flux, left_flux, top_flux, bottom_flux, total_flux;
    
    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        left_flux   = (xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)] 
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]))
            * 0.25 * dt;
        right_flux  = (xarea[THARR2D(1, 0, 1)]
            * (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)] 
            + xvel1[THARR2D(1, 0, 1)] + xvel1[THARR2D(1, 1, 1)]))
            * 0.25 * dt;

        bottom_flux = (yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)] 
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]))
            * 0.25 * dt;
        top_flux    = (yarea[THARR2D(0, 1, 0)]
            * (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)] 
            + yvel1[THARR2D(0, 1, 1)] + yvel1[THARR2D(1, 1, 1)]))
            * 0.25 * dt;

        total_flux = right_flux - left_flux + top_flux - bottom_flux;

        volume_change = volume[THARR2D(0, 0, 0)]
            / (volume[THARR2D(0, 0, 0)] + total_flux);

        min_cell_volume =
            MIN(volume[THARR2D(0, 0, 0)] + total_flux,
            MIN(volume[THARR2D(0, 0, 0)] + right_flux - left_flux,
                volume[THARR2D(0, 0, 0)] + top_flux - bottom_flux));

        if (volume_change <= 0.0)
        {
            err_condition_shared[threadIdx.x] = 1;
        }
        if (min_cell_volume <= 0.0)
        {
            err_condition_shared[threadIdx.x] = 2;
        }

        recip_volume = 1.0/volume[THARR2D(0, 0, 0)];

        energy_change = ((pressure[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)])
            + (viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)]))
            * total_flux * recip_volume;

        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)] - energy_change;
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)] * volume_change;
    }

    Reduce< BLOCK_SZ/2 >::run(err_condition_shared, error_condition, max_func);
}

extern "C" void pdv_kernel_cuda_
(int *error_condition, int *prdct, double *dtbyt)
{
    chunk.PdV_kernel(error_condition, *prdct, *dtbyt);
}

void CloverleafCudaChunk::PdV_kernel
(int* error_condition, int predict, double dt)
{
    CUDA_BEGIN_PROFILE;

    if (predict)
    {
        device_PdV_cuda_kernel_predict<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, pdv_reduce_array,
            xarea, yarea, volume, density0, density1,
            energy0, energy1, pressure, viscosity,
            xvel0, yvel0, xvel1, yvel1);
        CUDA_ERR_CHECK;
    }
    else
    {
        device_PdV_cuda_kernel_not_predict<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, dt, pdv_reduce_array,
            xarea, yarea, volume, density0, density1,
            energy0, energy1, pressure, viscosity,
            xvel0, yvel0, xvel1, yvel1);
        CUDA_ERR_CHECK;
    }

    *error_condition = *thrust::max_element(reduce_pdv,
                                            reduce_pdv + num_blocks);

    CUDA_END_PROFILE;

    if (1 == *error_condition)
    {
        std::cerr << "Negative volume in PdV kernel" << std::endl;
    }
    else if (2 == *error_condition)
    {
        std::cerr << "Negative cell volume in PdV kernel" << std::endl;
    }
}

