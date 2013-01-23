
#include "cuda_common.cu"
#include "ftocmacros.h"
#include <algorithm>

#include "chunk_cuda.cu"

extern CloverleafCudaChunk chunk;

__global__ void device_calc_dt_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,

const double g_small,
const double g_big,
const double dtmin,
const double dtc_safe,
const double dtu_safe,
const double dtv_safe,
const double dtdiv_safe,

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

    typedef double reduce_t;

    __shared__ reduce_t dt_min_shared[BLOCK_SZ];
    __shared__ reduce_t jk_ctrl_shared[BLOCK_SZ];
    dt_min_shared[threadIdx.x] = dt_min_val;
    jk_ctrl_shared[threadIdx.x] = jk_control;

    if(row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        dsx = celldx[column];
        dsy = celldy[row];

        cc = soundspeed[THARR2D(0, 0, 0)] * soundspeed[THARR2D(0, 0, 0)];
        cc += 2.0 * viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)];
        cc = MAX(sqrt(cc), g_small);

        dtct = dtc_safe * MIN(dsx, dsy)/cc;

        div = 0.0;

        //x
        dv1 = (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)])
            * xarea[THARR2D(0, 0, 1)];
        dv2 = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])
            * xarea[THARR2D(1, 0, 1)];

        div += dv2 - dv1;

        dtut = dtu_safe * 2.0 * volume[THARR2D(0, 0, 0)]
            / MAX(g_small*volume[THARR2D(0, 0, 0)], 
            MAX(fabs(dv1), fabs(dv2)));

        //y
        dv1 = (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)])
            * yarea[THARR2D(0, 0, 0)];
        dv2 = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])
            * yarea[THARR2D(0, 1, 0)];

        div += dv2 - dv1;

        dtvt = dtv_safe * 2.0 * volume[THARR2D(0, 0, 0)]
            / MAX(g_small*volume[THARR2D(0, 0, 0)], 
            MAX(fabs(dv1), fabs(dv2)));

        //
        div /= (2.0 * volume[THARR2D(0, 0, 0)]);

        dtdivt = (div < (-g_small)) ? dtdiv_safe * (-1.0/div) : g_big;

        dt_min_shared[threadIdx.x] = MIN(dtdivt, MIN(dtvt, MIN(dtct, dtvt)));

        //jk_ctrl_shared[threadIdx.x] = jk_control;
        //jk_ctrl_shared=(j+(x_max)*(k-1))+0.4_8
        jk_ctrl_shared[threadIdx.x] = (column + (x_max * (row - 1))) + 0.4;
    }

    Reduce< BLOCK_SZ/2 >::run(dt_min_shared, dt_min_out, min_func);
    Reduce< BLOCK_SZ/2 >::run(jk_ctrl_shared, jk_ctrl_out, max_func);

    /*
    __syncthreads();
    for(int offset = BLOCK_SZ / 2; offset > 0; offset /= 2)
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

    dt_min_out[blockIdx.x] = dt_min_shared[0];
    jk_ctrl_out[blockIdx.x] = jk_ctrl_shared[0];
    */
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
double* dt_min,
double* dt_min_val,
int* dtl_control,
double* xl_pos,
double* yl_pos,
int* jldt,
int* kldt,
int* small)
{
    chunk.calc_dt_kernel(*g_small, *g_big, *dtmin, *dtc_safe, *dtu_safe,
        *dtv_safe, *dtdiv_safe, dt_min_val, dtl_control, xl_pos, yl_pos,
        jldt, kldt, small);
}

static double taken = 0.0;

void CloverleafCudaChunk::calc_dt_kernel
(double g_small, double g_big, double dtmin,
double dtc_safe, double dtu_safe, double dtv_safe,
double dtdiv_safe, double* dt_min_val, int* dtl_control,
double* xl_pos, double* yl_pos, int* jldt, int* kldt, int* small)
{
    _CUDA_BEGIN_PROFILE_name(device);

    device_calc_dt_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min, x_max, y_min, y_max, g_small, g_big, dtmin, dtc_safe,
        dtu_safe, dtv_safe, dtdiv_safe, xarea, yarea, celldx, celldy,
        volume, density0, viscosity, soundspeed, xvel0, yvel0,
        work_array_1, work_array_2);
    errChk(__LINE__, __FILE__);

    // reduce_ptr 2 is a thrust wrapper around work_array_2
    *dt_min_val = *thrust::min_element(reduce_ptr_2, reduce_ptr_2 + num_blocks);

    // ditto on reduce ptr 1
    double jk_control = *thrust::max_element(reduce_ptr_1, reduce_ptr_1 + num_blocks);

    _CUDA_END_PROFILE_name(device);

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

