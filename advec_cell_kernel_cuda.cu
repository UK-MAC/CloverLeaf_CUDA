
#include <iostream>
#include "ftocmacros.h"
#include "advec_cell_cuda_kernels.cu"
#include "cuda_common.cu"
#include "omp.h"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

void advec_cell_cuda
(const int x_min, const int x_max, const int y_min, const int y_max,
const int dr,
const int swp_nmbr,
const double* vertexdx,
const double* vertexdy,
const double* volume,
double* density1,
double* energy1,
double* mass_flux_x,
const double* vol_flux_x,
double* mass_flux_y,
const double* vol_flux_y)
{

    pointer_storage.setSize(x_max, y_max);

    double* pre_vol_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* post_vol_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* ener_flux_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    double* energy1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, energy1, BUFSZ2D(0, 0));
    double* density1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density1, BUFSZ2D(0, 0));

    double* vol_flux_x_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vol_flux_x, BUFSZ2D(1, 0));
    double* vol_flux_y_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vol_flux_y, BUFSZ2D(0, 1));
    double* volume_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, volume, BUFSZ2D(0, 1));

    if(dr == 1)
    {
        //buffer size to send vol_flux x, mass_flux_x, and to allocate on device
        double* mass_flux_x_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
        double* vertexdx_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vertexdx, (x_max+5)*sizeof(double));

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

        device_pre_vol_kernel_x<<< (BUFSZ2D(0, 0)/BLOCK_SZ)/sizeof(double), BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            pre_vol_d,
            post_vol_d,
            volume_d,
            vol_flux_x_d,
            vol_flux_y_d
        );

        device_ener_flux_kernel_x<<< (BUFSZ2D(0, 0)/BLOCK_SZ)/sizeof(double), BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            volume_d,
            vol_flux_x_d,
            vol_flux_y_d,
            pre_vol_d,
            density1_d,
            energy1_d,
            ener_flux_d,
            vertexdx_d,
            mass_flux_x_d
        );

        device_advec_cell_kernel_x<<< (BUFSZ2D(0, 0)/BLOCK_SZ)/sizeof(double), BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            volume_d,
            vol_flux_x_d,
            vol_flux_y_d,
            pre_vol_d,
            density1_d,
            energy1_d,
            ener_flux_d,
            mass_flux_x_d
        );

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

        pointer_storage.freeDevStorage(vertexdx_d);
        pointer_storage.freeDevStorageAndCopy(mass_flux_x_d, mass_flux_x, BUFSZ2D(1, 0));
    }
    else if(dr == 2)
    {
        double* mass_flux_y_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
        double* vertexdy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vertexdy, (y_max+5)*sizeof(double));

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

        device_pre_vol_kernel_y<<< (BUFSZ2D(0, 0)/BLOCK_SZ)/sizeof(double), BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            pre_vol_d,
            post_vol_d,
            volume_d,
            vol_flux_x_d,
            vol_flux_y_d
        );

        device_ener_flux_kernel_y<<< (BUFSZ2D(0, 0)/BLOCK_SZ)/sizeof(double), BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            volume_d,
            vol_flux_x_d,
            vol_flux_y_d,
            pre_vol_d,
            density1_d,
            energy1_d,
            ener_flux_d,
            vertexdy_d,
            mass_flux_y_d
        );

        device_advec_cell_kernel_y<<< (BUFSZ2D(0, 0)/BLOCK_SZ)/sizeof(double), BLOCK_SZ >>>
        (
            x_min, x_max, y_min, y_max, swp_nmbr,
            volume_d,
            vol_flux_x_d,
            vol_flux_y_d,
            pre_vol_d,
            density1_d,
            energy1_d,
            ener_flux_d,
            mass_flux_y_d
        );

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

errChk(__LINE__, __FILE__);

        pointer_storage.freeDevStorage(vertexdy_d);
        pointer_storage.freeDevStorageAndCopy(mass_flux_y_d, mass_flux_y, BUFSZ2D(0, 1));
    }

    pointer_storage.freeDevStorage(pre_vol_d);
    pointer_storage.freeDevStorage(post_vol_d);
    pointer_storage.freeDevStorage(ener_flux_d);

    pointer_storage.freeDevStorage(volume_d);
    pointer_storage.freeDevStorage(vol_flux_x_d);
    pointer_storage.freeDevStorage(vol_flux_y_d);

    pointer_storage.freeDevStorageAndCopy(density1_d, density1, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(energy1_d, energy1, BUFSZ2D(0, 0));

}

extern "C" void advec_cell_kernel_cuda_
(const int* xmin, const int* xmax, const int* ymin, const int* ymax,
const int* dr,
const int* swp_nmbr,
const double* vertexdx,
const double* vertexdy,
const double* volume,
double* density1,
double* energy1,
double* mass_flux_x,
const double* vol_flux_x,
double* mass_flux_y,
const double* vol_flux_y)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    advec_cell_cuda(*xmin, *xmax, *ymin, *ymax, *dr, *swp_nmbr,
        vertexdx, vertexdy, volume, density1, energy1, mass_flux_x,
        vol_flux_x, mass_flux_y, vol_flux_y);
    #else
    chunk.advec_cell_kernel(*dr, *swp_nmbr);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::advec_cell_kernel
(int dr, int swp_nmbr)
{

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
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
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}

