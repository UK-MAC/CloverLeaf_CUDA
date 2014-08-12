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
 *  @brief CUDA timestep kernel
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Calculates the minimum timestep on the mesh chunk based on the CFL
 *  condition, the velocity gradient and the velocity divergence. A safety
 *  factor is used to ensure numerical stability.
 */

#include "cuda_common.hpp"
#include "kernel_files/calc_dt_kernel.cuknl"

extern "C" void calc_dt_kernel_cuda_
(double* g_small,
double* g_big,
double* dtmin,
double* dtc_safe,
double* dtu_safe,
double* dtv_safe,
double* dtdiv_safe,
double* dt_min_val,
int* dtl_control,
double* xl_pos,
double* yl_pos,
int* jldt,
int* kldt,
int* small)
{
    cuda_chunk.calc_dt_kernel(*g_small, *g_big, *dtmin, *dtc_safe, *dtu_safe,
        *dtv_safe, *dtdiv_safe, dt_min_val, dtl_control, xl_pos, yl_pos,
        jldt, kldt, small);
}

void CloverleafCudaChunk::calc_dt_kernel
(double g_small, double g_big, double dtmin,
double dtc_safe, double dtu_safe, double dtv_safe, double dtdiv_safe,
double* dt_min_val,
int* dtl_control,
double* xl_pos,
double* yl_pos,
int* jldt,
int* kldt,
int* small)
{
    CUDALAUNCH(device_calc_dt_kernel_cuda, g_small, g_big, dtmin, dtc_safe,
        dtu_safe, dtv_safe, dtdiv_safe, xarea, yarea, celldx, celldy,
        volume, density0, viscosity, soundspeed, xvel0, yvel0,
        reduce_buf_1, reduce_buf_2);

    // reduce_ptr 2 is a thrust wrapper around work_array_2
    *dt_min_val = *thrust::min_element(reduce_ptr_2,
                                       reduce_ptr_2 + num_blocks);

    // ditto on reduce ptr 1
    double jk_control = *thrust::max_element(reduce_ptr_1,
                                             reduce_ptr_1 + num_blocks);

    *dtl_control = 10.01 * (jk_control - static_cast<int>(jk_control));

    jk_control = jk_control - (jk_control - static_cast<int>(jk_control));
    int tmp_jldt = *jldt = (static_cast<int>(jk_control)) % x_max;
    int tmp_kldt = *kldt = 1 + (jk_control/x_max);

    *xl_pos = thr_cellx[tmp_jldt];
    *yl_pos = thr_celly[tmp_kldt];

    *small = (*dt_min_val < dtmin) ? 1 : 0;

    if (*small != 0)
    {
        std::cerr << "Timestep information:" << std::endl;
        std::cerr << "j, k     : " << tmp_jldt << " " << tmp_kldt << std::endl;
        std::cerr << "x, y     : " << thr_cellx[tmp_jldt] << " " << thr_celly[tmp_kldt] << std::endl;
        std::cerr << "timestep : " << *dt_min_val << std::endl;
        std::cerr << "Cell velocities;" << std::endl;
        std::cerr << thr_xvel0[tmp_jldt  +(x_max+5)*tmp_kldt  ] << "\t";
        std::cerr << thr_yvel0[tmp_jldt  +(x_max+5)*tmp_kldt  ] << std::endl;
        std::cerr << thr_xvel0[tmp_jldt+1+(x_max+5)*tmp_kldt  ] << "\t";
        std::cerr << thr_yvel0[tmp_jldt+1+(x_max+5)*tmp_kldt  ] << std::endl;
        std::cerr << thr_xvel0[tmp_jldt+1+(x_max+5)*(tmp_kldt+1)] << "\t";
        std::cerr << thr_yvel0[tmp_jldt+1+(x_max+5)*(tmp_kldt+1)] << std::endl;
        std::cerr << thr_xvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << "\t";
        std::cerr << thr_yvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << std::endl;
        std::cerr << "density, energy, pressure, soundspeed " << std::endl;
        std::cerr << thr_density0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        std::cerr << thr_energy0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        std::cerr << thr_pressure[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        std::cerr << thr_soundspeed[tmp_jldt+(x_max+5)*tmp_kldt] << std::endl;
    }
}

