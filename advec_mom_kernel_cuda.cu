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
 *  @brief CUDA momentum advection driver
 *  @author Michael Boulton NVIDIA Corporation
 *  @details CUDA momentum advection driver.
 */

#include "cuda_common.hpp"
#include "kernel_files/advec_mom_kernel.cuknl"

extern "C" void advec_mom_kernel_cuda_
(int *whch_vl,
int *swp_nmbr,
int *drctn)
{
    cuda_chunk.advec_mom_kernel(*whch_vl, *swp_nmbr, *drctn);
}

void CloverleafCudaChunk::advec_mom_kernel
(int which_vel, int sweep_number, int direction)
{
    int mom_sweep = direction + (2 * (sweep_number - 1));

    CUDALAUNCH(device_advec_mom_vol_kernel_cuda,
        mom_sweep, work_array_1, work_array_2, volume,
        vol_flux_x, vol_flux_y);

    /*
    post_vol = work array 1
    node_flux = work array 2
    node_mass_post = work array 3
    node_mass_pre = work array 4
    mom_flux = work array 5
    */

    double* vel1;
    if (which_vel == 1)
    {
        vel1 =  xvel1;
    }
    else
    {
        vel1 =  yvel1;
    }

    if (direction == 1)
    {
        CUDALAUNCH(device_advec_mom_node_flux_post_x_kernel_cuda,
            work_array_2, work_array_3, mass_flux_x,
            work_array_1, density1);

        CUDALAUNCH(device_advec_mom_node_pre_x_kernel_cuda,
            work_array_2, work_array_3, work_array_4);

        CUDALAUNCH(device_advec_mom_flux_x_kernel_cuda,
            work_array_2, work_array_3, work_array_4,
            vel1, celldx, work_array_5);

        CUDALAUNCH(device_advec_mom_xvel_kernel_cuda,
            work_array_3, work_array_4, work_array_5,
            vel1);
    }
    else if (direction == 2)
    {
        CUDALAUNCH(device_advec_mom_node_flux_post_y_kernel_cuda,
            work_array_2, work_array_3, mass_flux_y,
            work_array_1, density1);

        CUDALAUNCH(device_advec_mom_node_pre_y_kernel_cuda,
            work_array_2, work_array_3, work_array_4);

        CUDALAUNCH(device_advec_mom_flux_y_kernel_cuda,
            work_array_2, work_array_3, work_array_4,
            vel1, celldy, work_array_5);

        CUDALAUNCH(device_advec_mom_yvel_kernel_cuda,
            work_array_3, work_array_4, work_array_5,
            vel1);
    }
}

