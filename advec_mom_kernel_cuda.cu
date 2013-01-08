
#include <iostream>
#include "ftocmacros.h"
#include "omp.h"
#include "cuda_common.cu"
#include "advec_mom_cuda_kernels.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

void advec_mom_cuda
(int x_min,int x_max,int y_min,int y_max,

      double *xvel1,
      double *yvel1,
const double *mass_flux_x,
const double *vol_flux_x,
const double *mass_flux_y,
const double *vol_flux_y,
const double *volume,
const double *density1,
const double *celldx,
const double *celldy,

int which_vel,
int sweep_number,
int direction)
{

    pointer_storage.setSize(x_max, y_max);

    double* volume_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, volume, BUFSZ2D(0, 0));
    double* density1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density1, BUFSZ2D(0, 0));
    double* vol_flux_x_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vol_flux_x, BUFSZ2D(1, 0));
    double* vol_flux_y_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vol_flux_y, BUFSZ2D(0, 1));

    double* mom_flux_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* post_vol_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* node_flux_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* node_mass_post_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* node_mass_pre_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* pre_vol_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

    int mom_sweep = direction + (2 * (sweep_number - 1));

    device_advec_mom_vol_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, mom_sweep, post_vol_d, pre_vol_d, volume_d, vol_flux_x_d, vol_flux_y_d);
errChk(__LINE__, __FILE__);

    //not used again
    pointer_storage.freeDevStorage(vol_flux_x_d);
    pointer_storage.freeDevStorage(vol_flux_y_d);

    double* vel1;
    if(which_vel == 1)
    {
        vel1 =  xvel1;
    }
    else
    {
        vel1 =  yvel1;
    }

    if(direction == 1)
    {
//*
        double* xvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vel1, BUFSZ2D(1, 1));
        double* mass_flux_x_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, mass_flux_x, BUFSZ2D(1, 0));
        double* celldx_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celldx, (x_max+4)*sizeof(double));

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

        device_advec_mom_node_flux_post_x_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_flux_d, node_mass_post_d, mass_flux_x_d, post_vol_d, density1_d);
errChk(__LINE__, __FILE__);

        device_advec_mom_node_pre_x_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_flux_d, node_mass_post_d, node_mass_pre_d);
errChk(__LINE__, __FILE__);

        device_advec_mom_flux_x_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_flux_d, node_mass_post_d, node_mass_pre_d, xvel1_d, celldx_d, mom_flux_d);
errChk(__LINE__, __FILE__);

        device_advec_mom_xvel_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_mass_post_d, node_mass_pre_d, mom_flux_d, xvel1_d);
errChk(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

        pointer_storage.freeDevStorageAndCopy(xvel1_d, vel1, BUFSZ2D(1, 1));
        pointer_storage.freeDevStorage(mass_flux_x_d);
        pointer_storage.freeDevStorage(celldx_d);
// */
    }
    else
    {
//*
        double* yvel1_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, vel1, BUFSZ2D(1, 1));
        double* mass_flux_y_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, mass_flux_y, BUFSZ2D(0, 1));
        double* celldy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celldy, (y_max+4)*sizeof(double));

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

        device_advec_mom_node_flux_post_y_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_flux_d, node_mass_post_d, mass_flux_y_d, post_vol_d, density1_d);
errChk(__LINE__, __FILE__);

        device_advec_mom_node_pre_y_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_flux_d, node_mass_post_d, node_mass_pre_d);
errChk(__LINE__, __FILE__);

        device_advec_mom_flux_y_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_flux_d, node_mass_post_d, node_mass_pre_d, yvel1_d, celldy_d, mom_flux_d);
errChk(__LINE__, __FILE__);

        device_advec_mom_yvel_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, node_mass_post_d, node_mass_pre_d, mom_flux_d, yvel1_d);
errChk(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

        pointer_storage.freeDevStorageAndCopy(yvel1_d, vel1, BUFSZ2D(1, 1));
        pointer_storage.freeDevStorage(mass_flux_y_d);
        pointer_storage.freeDevStorage(celldy_d);
// */
    }

    pointer_storage.freeDevStorage(mom_flux_d);
    pointer_storage.freeDevStorage(post_vol_d);
    pointer_storage.freeDevStorage(node_flux_d);
    pointer_storage.freeDevStorage(node_mass_post_d);
    pointer_storage.freeDevStorage(node_mass_pre_d);
    pointer_storage.freeDevStorage(pre_vol_d);

    pointer_storage.freeDevStorage(volume_d);
    pointer_storage.freeDevStorage(density1_d);

}

extern "C" void advec_mom_kernel_cuda_
(int *xmin,int *xmax,int *ymin,int *ymax,
      double *xvel1,
      double *yvel1,
const double *mass_flux_x,
const double *vol_flux_x,
const double *mass_flux_y,
const double *vol_flux_y,
const double *volume,
const double *density1,
const double *celldx,
const double *celldy,
int *whch_vl,
int *swp_nmbr,
int *drctn)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    advec_mom_cuda(*xmin, *xmax, *ymin, *ymax,
        xvel1, yvel1, mass_flux_x, vol_flux_x, mass_flux_y,
        vol_flux_y, volume, density1, celldx, celldy,
        *whch_vl, *swp_nmbr, *drctn);
    #else
    chunk.advec_mom_kernel(*whch_vl, *swp_nmbr, *drctn);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::advec_mom_kernel
(int which_vel, int sweep_number, int direction)
{
    int mom_sweep = direction + (2 * (sweep_number - 1));

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif

    device_advec_mom_vol_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, mom_sweep, work_array_1, work_array_2, volume, vol_flux_x, vol_flux_y);
    errChk(__LINE__, __FILE__);

    /*
    post_vol = work array 1
    node_flux = work array 2
    node_mass_post = work array 3
    node_mass_pre = work array 4
    mom_flux = work array 5
    */

    double* vel1;
    if(which_vel == 1)
    {
        vel1 =  xvel1;
    }
    else
    {
        vel1 =  yvel1;
    }

    if(direction == 1)
    {
        device_advec_mom_node_flux_post_x_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, work_array_3, mass_flux_x, work_array_1, density1);
        errChk(__LINE__, __FILE__);

        device_advec_mom_node_pre_x_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, work_array_3, work_array_4);
        errChk(__LINE__, __FILE__);

        device_advec_mom_flux_x_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, work_array_3, work_array_4, vel1, celldx, work_array_5);
        errChk(__LINE__, __FILE__);

        device_advec_mom_xvel_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_3, work_array_4, work_array_5, vel1);
        errChk(__LINE__, __FILE__);
    }
    else if (direction == 2)
    {
        device_advec_mom_node_flux_post_y_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, work_array_3, mass_flux_y, work_array_1, density1);
        errChk(__LINE__, __FILE__);

        device_advec_mom_node_pre_y_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, work_array_3, work_array_4);
        errChk(__LINE__, __FILE__);

        device_advec_mom_flux_y_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, work_array_3, work_array_4, vel1, celldy, work_array_5);
        errChk(__LINE__, __FILE__);

        device_advec_mom_yvel_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_3, work_array_4, work_array_5, vel1);
        errChk(__LINE__, __FILE__);
    }

#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
}

