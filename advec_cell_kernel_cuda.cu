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
 *  @brief CUDA cell advection kernel driver.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details CUDA cell advection kernel driver.
 */

#include "cuda_common.hpp"
#include "kernel_files/advec_cell_kernel.cuknl"

extern "C" void advec_cell_kernel_cuda_
(const int* dr,
 const int* swp_nmbr)
{
    cuda_chunk.advec_cell_kernel(*dr, *swp_nmbr);
}

void CloverleafCudaChunk::advec_cell_kernel
(int dr, int swp_nmbr)
{
    if (dr == 1)
    {
        CUDALAUNCH(device_pre_vol_kernel_x, swp_nmbr,
            work_array_1,
            work_array_2,
            volume,
            vol_flux_x,
            vol_flux_y
        );

        CUDALAUNCH(device_ener_flux_kernel_x, swp_nmbr,
            volume,
            vol_flux_x,
            vol_flux_y,
            work_array_1,
            density1,
            energy1,
            work_array_2,
            vertexdx,
            mass_flux_x
        );

        CUDALAUNCH(device_advec_cell_kernel_x, swp_nmbr,
            volume,
            vol_flux_x,
            vol_flux_y,
            work_array_1,
            density1,
            energy1,
            work_array_2,
            mass_flux_x
        );
    }
    else if (dr == 2)
    {
        CUDALAUNCH(device_pre_vol_kernel_y, swp_nmbr,
            work_array_1,
            work_array_2,
            volume,
            vol_flux_x,
            vol_flux_y
        );

        CUDALAUNCH(device_ener_flux_kernel_y, swp_nmbr,
            volume,
            vol_flux_x,
            vol_flux_y,
            work_array_1,
            density1,
            energy1,
            work_array_2,
            vertexdy,
            mass_flux_y
        );

        CUDALAUNCH(device_advec_cell_kernel_y, swp_nmbr,
            volume,
            vol_flux_x,
            vol_flux_y,
            work_array_1,
            density1,
            energy1,
            work_array_2,
            mass_flux_y
        );
    }
}

