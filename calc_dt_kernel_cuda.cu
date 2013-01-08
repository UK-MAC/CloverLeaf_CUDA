
#include "cuda_common.cu"
#include "ftocmacros.h"
#include <algorithm>

#include "chunk_cuda.cu"

extern CudaDevPtrStorage pointer_storage;

extern CloverleafCudaChunk chunk;

__global__ void device_calc_dt_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,

double g_small,
double g_big,
double dtmin,
double dtc_safe,
double dtu_safe,
double dtv_safe,
double dtdiv_safe,

const double* __restrict const xarea,
const double* __restrict const yarea,
const double* __restrict const celldx,
const double* __restrict const celldy,
const double* __restrict const volume,
const double* __restrict const density0,
const double* __restrict const viscosity,
const double* __restrict const soundspeed,
const double* __restrict const xvel0,
const double* __restrict const yvel0,

double* __restrict const jk_ctrl_out,
double* __restrict const dt_min_out)
{
    __kernel_indexes;

    double dsx, dsy;
    double cc;
    double dtct;
    double div;
    double dv1;
    double dv2;
    double dtut;
    double dtvt;
    double dtdivt;

    //reduced
    double dt_min_val = g_big;
    double jk_control = 0.0;

    __shared__ double dt_min_shared[BLOCK_SZ];
    __shared__ double jk_ctrl_shared[BLOCK_SZ];
    dt_min_shared[threadIdx.x] = dt_min_val;
    jk_ctrl_shared[threadIdx.x] = jk_control;

    if(row > 1 && column > 1
    && row < y_max+2 && column < x_max+2)
    {
        dsx = celldx[column];
        dsy = celldy[row];

        cc = soundspeed[THARR2D(0, 0, 0)] * soundspeed[THARR2D(0, 0, 0)];
        cc += 2.0 * viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)];
        cc = MAX(sqrt(cc), g_small);

        dtct = MIN(dsx, dsy)/cc;

        div = 0.0;

        //x
        dv1 = (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)])
            * xarea[THARR2D(0, 0, 1)];
        dv2 = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])
            * xarea[THARR2D(1, 0, 1)];

        div += dv2 - dv1;

        dtut = 2.0 * volume[THARR2D(0, 0, 0)]
            / MAX(g_small*volume[THARR2D(0, 0, 0)], 
            MAX(fabs(dv1), fabs(dv2)));

        //y
        dv1 = (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)])
            * yarea[THARR2D(0, 0, 0)];
        dv2 = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])
            * yarea[THARR2D(0, 1, 0)];

        div += dv2 - dv1;

        dtvt = 2.0 * volume[THARR2D(0, 0, 0)]
            / MAX(g_small*volume[THARR2D(0, 0, 0)], 
            MAX(fabs(dv1), fabs(dv2)));

        //
        div /= (2.0 * volume[THARR2D(0, 0, 0)]);

        dtdivt = (div < (-g_small)) ? dtdivt = - (1.0/div) : dtdivt = g_big;

        //1
        if(dtct * dtc_safe < dt_min_val)
        {
            jk_control = (column + x_max * (row - 1)) + 0.1;
            dt_min_val = dtct * dtc_safe;
        }

        //2
        if(dtut * dtu_safe < dt_min_val)
        {
            jk_control = (column + x_max * (row - 1)) + 0.2;
            dt_min_val = dtut * dtu_safe;
        }

        //3
        if(dtvt * dtv_safe < dt_min_val)
        {
            jk_control = (column + x_max * (row - 1)) + 0.3;
            dt_min_val = dtvt * dtv_safe;
        }

        //4
        if(dtdivt * dtdiv_safe < dt_min_val)
        {
            jk_control = (column + x_max * (row - 1)) + 0.4;
            dt_min_val = dtdivt * dtdiv_safe;
        }

        dt_min_shared[threadIdx.x] = dt_min_val;
        jk_ctrl_shared[threadIdx.x] = jk_control;

    }

    __syncthreads();
    //*
    for(size_t offset = BLOCK_SZ / 2; offset > 0; offset /= 2)
    {
        if(threadIdx.x < offset)
        {
            dt_min_shared[threadIdx.x] = MIN(dt_min_shared[threadIdx.x],
                dt_min_shared[threadIdx.x + offset]);
            jk_ctrl_shared[threadIdx.x] = MAX(jk_ctrl_shared[threadIdx.x],
                jk_ctrl_shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    // */

    dt_min_out[blockIdx.x] = dt_min_shared[0];
    jk_ctrl_out[blockIdx.x] = jk_ctrl_shared[0];
}

void calc_dt_cuda
(int x_min, int x_max, int y_min, int y_max,

double g_small,
double g_big,
double dtmin,
double dtc_safe,
double dtu_safe,
double dtv_safe,
double dtdiv_safe,

double* xarea,
double* yarea,
double* cellx,
double* celly,
double* celldx,
double* celldy,
double* volume,
double* density0,
double* energy0,
double* pressure,
double* viscosity,
double* soundspeed,
double* xvel0,
double* yvel0,

double* dt_min_val,
int* dtl_control,
double* xl_pos,
double* yl_pos,
int* jldt,
int* kldt,
int* small)
{
    pointer_storage.setSize(x_max, y_max);

    double* xarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xarea, BUFSZ2D(1, 0));
    double* yarea_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yarea, BUFSZ2D(0, 1));
    double* celldx_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celldx, (x_max+4)*sizeof(double));
    double* celldy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, celldy, (y_max+4)*sizeof(double));
    double* density0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density0, BUFSZ2D(0, 0));
    double* viscosity_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, viscosity, BUFSZ2D(0, 0));
    double* soundspeed_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, soundspeed, BUFSZ2D(0, 0));
    double* xvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, xvel0, BUFSZ2D(1, 1));
    double* yvel0_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, yvel0, BUFSZ2D(1, 1));
    double* volume_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, volume, BUFSZ2D(0, 0));

    size_t num_blocks = (((x_max+4)*(y_max+4))/BLOCK_SZ);
    //dt_min_val
    thrust::device_ptr<double> reduce_ptr_1 =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* dt_min_d = thrust::raw_pointer_cast(reduce_ptr_1);

    //jk_control
    thrust::device_ptr<double> reduce_ptr_2 =
        thrust::device_malloc<double>(num_blocks*sizeof(double));
    double* jk_ctrl_d = thrust::raw_pointer_cast(reduce_ptr_2);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_calc_dt_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, g_small, g_big, dtmin, dtc_safe,
        dtu_safe, dtv_safe, dtdiv_safe, xarea_d, yarea_d, celldx_d, celldy_d, volume_d, density0_d,
        viscosity_d, soundspeed_d, xvel0_d, yvel0_d, jk_ctrl_d, dt_min_d);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    cudaFree(xarea_d);
    cudaFree(yarea_d);
    cudaFree(celldx_d);
    cudaFree(celldy_d);
    cudaFree(density0_d);
    cudaFree(viscosity_d);
    cudaFree(soundspeed_d);
    cudaFree(xvel0_d);
    cudaFree(yvel0_d);
    cudaFree(volume_d);

    *dt_min_val = *thrust::min_element(reduce_ptr_1, reduce_ptr_1 + num_blocks);

    //l_control
    double jk_control = *thrust::max_element(reduce_ptr_2, reduce_ptr_2 + num_blocks);
    *dtl_control = 10.01 * (jk_control - (int)jk_control);

    jk_control = jk_control - (jk_control - (int)jk_control);
    int tmp_jldt = *jldt = ((int)jk_control) % x_max;
    int tmp_kldt = *kldt = 1 + (jk_control/x_max);
    * xl_pos = cellx[tmp_jldt];
    * yl_pos = celly[tmp_kldt];

    * small = (*dt_min_val < dtmin) ? 1 : 0;

    if(* small != 0)
    {
        std::cerr << "Timestep information:" << std::endl;
        std::cerr << "j, k     : " << tmp_jldt << tmp_kldt << std::endl;
        std::cerr << "x, y     : " << cellx[tmp_jldt] << celly[tmp_kldt] << std::endl;
        std::cerr << "timestep : " << *dt_min_val << std::endl;
        std::cerr << "Cell velocities;" << std::endl;
        std::cerr << xvel0[tmp_jldt  +(x_max+5)*tmp_kldt  ] << "\t";
        std::cerr << yvel0[tmp_jldt  +(x_max+5)*tmp_kldt  ] << std::endl;
        std::cerr << xvel0[tmp_jldt+1+(x_max+5)*tmp_kldt  ] << "\t";
        std::cerr << yvel0[tmp_jldt+1+(x_max+5)*tmp_kldt  ] << std::endl;
        std::cerr << xvel0[tmp_jldt+1+(x_max+5)*(tmp_kldt+1)] << "\t";
        std::cerr << yvel0[tmp_jldt+1+(x_max+5)*(tmp_kldt+1)] << std::endl;
        std::cerr << xvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << "\t";
        std::cerr << yvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << std::endl;
        std::cerr << "density, energy, pressure, soundspeed " << std::endl;
        std::cerr << density0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        std::cerr << energy0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        std::cerr << pressure[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        std::cerr << soundspeed[tmp_jldt+(x_max+5)*tmp_kldt] << std::endl;
    }

    thrust::device_free(reduce_ptr_2);
    thrust::device_free(reduce_ptr_1);

}

extern "C" void calc_dt_kernel_cuda_
(int* xmin, int* xmax, int* ymin, int* ymax,

double* g_small,
double* g_big,
double* dtmin,
double* dtc_safe,
double* dtu_safe,
double* dtv_safe,
double* dtdiv_safe,

double* xarea,
double* yarea,
double* cellx,
double* celly,
double* celldx,
double* celldy,
double* volume,
double* density0,
double* energy0,
double* pressure,
double* viscosity,
double* soundspeed,
double* xvel0,
double* yvel0,

//output
double* dt_min_val,
int* dtl_control,
double* xl_pos,
double* yl_pos,
int* jldt,
int* kldt,
int* small)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    calc_dt_cuda(* xmin, * xmax, * ymin, * ymax, * g_small,
        * g_big, * dtmin, * dtc_safe, * dtu_safe, * dtv_safe, *dtdiv_safe,
        xarea, yarea, cellx, celly, celldx, celldy, volume, density0,
        energy0, pressure, viscosity, soundspeed, xvel0, yvel0,
        dt_min_val, dtl_control, xl_pos, yl_pos, jldt, kldt, small);
    #else
    chunk.calc_dt_kernel(*g_small, *g_big, *dtmin, *dtc_safe, *dtu_safe,
        *dtv_safe, *dtdiv_safe, dt_min_val, dtl_control, xl_pos, yl_pos,
        jldt, kldt, small);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

void CloverleafCudaChunk::calc_dt_kernel
(double g_small, double g_big, double dtmin,
double dtc_safe, double dtu_safe, double dtv_safe,
double dtdiv_safe, double* dt_min_val, int* dtl_control,
double* xl_pos, double* yl_pos, int* jldt, int* kldt, int* small)
{

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_calc_dt_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, g_small, g_big, dtmin, dtc_safe,
        dtu_safe, dtv_safe, dtdiv_safe, xarea, yarea, celldx, celldy,
        volume, density0, viscosity, soundspeed, xvel0, yvel0,
        work_array_1, work_array_2);
    errChk(__LINE__, __FILE__);

    *dt_min_val = *thrust::min_element(reduce_ptr_2, reduce_ptr_2 + num_blocks);

    double jk_control = *thrust::max_element(reduce_ptr_1, reduce_ptr_1 + num_blocks);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif
    *dtl_control = 10.01 * (jk_control - (int)jk_control);

    jk_control = jk_control - (jk_control - (int)jk_control);
    int tmp_jldt = *jldt = ((int)jk_control) % x_max;
    int tmp_kldt = *kldt = 1 + (jk_control/x_max);

    * xl_pos = thr_cellx[tmp_jldt];
    * yl_pos = thr_celly[tmp_kldt];

    * small = (*dt_min_val < dtmin) ? 1 : 0;

    if(* small != 0)
    {
        std::cerr << "Timestep information:" << std::endl;
        std::cerr << "j, k     : " << tmp_jldt << tmp_kldt << std::endl;
        std::cerr << "x, y     : " << thr_cellx[tmp_jldt] << thr_celly[tmp_kldt] << std::endl;
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

