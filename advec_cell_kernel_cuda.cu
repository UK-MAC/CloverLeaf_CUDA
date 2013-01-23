
#include <iostream>
#include "ftocmacros.h"
#include "advec_cell_cuda_kernels.cu"
#include "cuda_common.cu"
#include "omp.h"

#include "chunk_cuda.cu"

extern CloverleafCudaChunk chunk;

extern "C" void advec_cell_kernel_cuda_
(const int* xmin, const int* xmax, const int* ymin, const int* ymax,
const int* dr,
const int* swp_nmbr,
const bool* vector,
const double* vertexdx,
const double* vertexdy,
const double* volume,
double* density1,
double* energy1,
double* mass_flux_x,
const double* vol_flux_x,
double* mass_flux_y,
const double* vol_flux_y,

double* unused_array1,
double* unused_array2,
double* unused_array3,
double* unused_array4,
double* unused_array5,
double* unused_array6,
double* unused_array7)
{
    chunk.advec_cell_kernel(*dr, *swp_nmbr);
}

void CloverleafCudaChunk::advec_cell_kernel
(int dr, int swp_nmbr)
{

_CUDA_BEGIN_PROFILE_name(device);
    if(dr == 1)
    {
        device_pre_vol_kernel_x<<< num_blocks, BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            work_array_1,
            work_array_2,
            volume,
            vol_flux_x,
            vol_flux_y
        );
        errChk(__LINE__, __FILE__);

        device_ener_flux_kernel_x<<< num_blocks, BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
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
        errChk(__LINE__, __FILE__);

        device_advec_cell_kernel_x<<< num_blocks, BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            volume,
            vol_flux_x,
            vol_flux_y,
            work_array_1,
            density1,
            energy1,
            work_array_2,
            mass_flux_x
        );
        errChk(__LINE__, __FILE__);
    }
    else if(dr == 2)
    {

        device_pre_vol_kernel_y<<< num_blocks, BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            work_array_1,
            work_array_2,
            volume,
            vol_flux_x,
            vol_flux_y
        );
        errChk(__LINE__, __FILE__);

        device_ener_flux_kernel_y<<< num_blocks, BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
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
        errChk(__LINE__, __FILE__);

        device_advec_cell_kernel_y<<< num_blocks, BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            volume,
            vol_flux_x,
            vol_flux_y,
            work_array_1,
            density1,
            energy1,
            work_array_2,
            mass_flux_y
        );

        errChk(__LINE__, __FILE__);
    }
_CUDA_END_PROFILE_name(device);
}

