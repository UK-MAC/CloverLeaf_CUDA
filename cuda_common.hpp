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
 *  @brief CUDA common file
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Contains common elements for cuda kernels
 */

#ifndef __CUDA_COMMON_INC
#define __CUDA_COMMON_INC

#include "thrust/device_allocator.h"
#include "thrust/extrema.h"
#include "kernel_files/cuda_kernel_header.hpp"
#include <map>
#include <vector>

// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density0      1
#define FIELD_density1      2
#define FIELD_energy0       3
#define FIELD_energy1       4
#define FIELD_pressure      5
#define FIELD_viscosity     6
#define FIELD_soundspeed    7
#define FIELD_xvel0         8
#define FIELD_xvel1         9
#define FIELD_yvel0         10
#define FIELD_yvel1         11
#define FIELD_vol_flux_x    12
#define FIELD_vol_flux_y    13
#define FIELD_mass_flux_x   14
#define FIELD_mass_flux_y   15
#define NUM_FIELDS          15

// which side to pack - keep the same as in fortran file
#define CHUNK_LEFT 1
#define CHUNK_left 1
#define CHUNK_RIGHT 2
#define CHUNK_right 2
#define CHUNK_BOTTOM 3
#define CHUNK_bottom 3
#define CHUNK_TOP 4
#define CHUNK_top 4
#define EXTERNAL_FACE       (-1)

#define CELL_DATA   1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4

#define INITIALISE_ARGS \
    /* values used to control operation */\
    int* in_x_min, \
    int* in_x_max, \
    int* in_y_min, \
    int* in_y_max, \
    bool* in_profiler_on

/*******************/

// disable checking for errors after kernel calls / memory allocation
#ifdef NO_ERR_CHK

// do nothing instead
#define CUDA_ERR_CHECK ;

#else

#include <iostream>

#define CUDA_ERR_CHECK errorHandler(__LINE__, __FILE__);

#endif //NO_ERR_CHK

/*******************/

// enormous ugly macro that profiles kernels + checks if there were any errors
#define CUDALAUNCH(funcname, ...)                               \
    if (profiler_on)                                            \
    {                                                           \
        cudaEventCreate(&_t0);                                  \
        cudaEventRecord(_t0);                                   \
    }                                                           \
    funcname<<<num_blocks, BLOCK_SZ>>>(x_min, x_max, y_min, y_max, __VA_ARGS__); \
    CUDA_ERR_CHECK;                                             \
    if (profiler_on)                                            \
    {                                                           \
        cudaEventCreate(&_t1);                                  \
        cudaEventRecord(_t1);                                   \
        cudaEventSynchronize(_t1);                              \
        cudaEventElapsedTime(&taken, _t0, _t1);                 \
        std::string func_name(#funcname);                       \
        if (kernel_times.end() != kernel_times.find(func_name)) \
        {                                                       \
            kernel_times.at(func_name) += taken;                \
        }                                                       \
        else                                                    \
        {                                                       \
            kernel_times[func_name] = taken;                    \
        }                                                       \
    }

/*******************/

typedef struct cell_info {
    const int x_extra;
    const int y_extra;
    const int x_invert;
    const int y_invert;
    const int x_face;
    const int y_face;
    const int grid_type;

    cell_info
    (int in_x_extra, int in_y_extra,
    int in_x_invert, int in_y_invert,
    int in_x_face, int in_y_face,
    int in_grid_type)
    :x_extra(in_x_extra), y_extra(in_y_extra),
    x_invert(in_x_invert), y_invert(in_y_invert),
    x_face(in_x_face), y_face(in_y_face),
    grid_type(in_grid_type)
    {
        ;
    }

} cell_info_t;

// types of array data
const static cell_info_t CELL(    0, 0,  1,  1, 0, 0, CELL_DATA);
const static cell_info_t VERTEX_X(1, 1, -1,  1, 0, 0, VERTEX_DATA);
const static cell_info_t VERTEX_Y(1, 1,  1, -1, 0, 0, VERTEX_DATA);
const static cell_info_t X_FACE(  1, 0, -1,  1, 1, 0, X_FACE_DATA);
const static cell_info_t Y_FACE(  0, 1,  1, -1, 0, 1, Y_FACE_DATA);

class CloverleafCudaChunk
{
private:
    // work arrays
    double* volume;
    double* soundspeed;
    double* pressure;
    double* viscosity;

    double* density0;
    double* density1;
    double* energy0;
    double* energy1;
    double* xvel0;
    double* xvel1;
    double* yvel0;
    double* yvel1;
    double* xarea;
    double* yarea;
    double* vol_flux_x;
    double* vol_flux_y;
    double* mass_flux_x;
    double* mass_flux_y;

    double* cellx;
    double* celly;
    double* celldx;
    double* celldy;
    double* vertexx;
    double* vertexy;
    double* vertexdx;
    double* vertexdy;

    // used in calc_dt to retrieve values
    thrust::device_ptr< double > thr_cellx;
    thrust::device_ptr< double > thr_celly;
    thrust::device_ptr< double > thr_xvel0;
    thrust::device_ptr< double > thr_yvel0;
    thrust::device_ptr< double > thr_xvel1;
    thrust::device_ptr< double > thr_yvel1;
    thrust::device_ptr< double > thr_density0;
    thrust::device_ptr< double > thr_energy0;
    thrust::device_ptr< double > thr_pressure;
    thrust::device_ptr< double > thr_soundspeed;

    // holding temporary stuff like post_vol etc.
    double* work_array_1;
    double* work_array_2;
    double* work_array_3;
    double* work_array_4;
    double* work_array_5;
    double* work_array_6;

    // buffers used in mpi transfers
    double* dev_left_send_buffer;
    double* dev_right_send_buffer;
    double* dev_top_send_buffer;
    double* dev_bottom_send_buffer;
    double* dev_left_recv_buffer;
    double* dev_right_recv_buffer;
    double* dev_top_recv_buffer;
    double* dev_bottom_recv_buffer;

    // holding temporary stuff like post_vol etc.
    double* reduce_buf_1;
    double* reduce_buf_2;
    double* reduce_buf_3;
    double* reduce_buf_4;
    double* reduce_buf_5;
    double* reduce_buf_6;

    // used for reductions in calc dt, pdv, field summary
    thrust::device_ptr< double > reduce_ptr_1;
    thrust::device_ptr< double > reduce_ptr_2;
    thrust::device_ptr< double > reduce_ptr_3;
    thrust::device_ptr< double > reduce_ptr_4;
    thrust::device_ptr< double > reduce_ptr_5;
    thrust::device_ptr< double > reduce_ptr_6;

    // number of blocks for work space
    unsigned int num_blocks;

    //as above, but for pdv kernel only
    int* pdv_reduce_array;
    thrust::device_ptr< int > reduce_pdv;

    // values used to control operation
    int x_min;
    int x_max;
    int y_min;
    int y_max;

    // if being profiled
    bool profiler_on;
    // for recording times if profiling is on
    std::map<std::string, double> kernel_times;
    // events used for timing
    float taken;
    cudaEvent_t _t0, _t1;

    // mpi packing
    #define PACK_ARGS                                       \
        int chunk_1, int chunk_2, int external_face,        \
        int x_inc, int y_inc, int depth, int which_field,   \
        double *buffer_1, double *buffer_2
    int getBufferSize
    (int edge, int depth, int x_inc, int y_inc);

    void unpackBuffer
    (const int which_array,
    const int which_side,
    double* buffer,
    const int buffer_size,
    const int depth);

    void packBuffer
    (const int which_array,
    const int which_side,
    double* buffer,
    const int buffer_size,
    const int depth);

    void errorHandler
    (int line_num, const char* file);

    void update_array
    (int x_min, int x_max, int y_min, int y_max,
     cell_info_t const& grid_type,
     const int* chunk_neighbours,
     double* cur_array_d,
     int depth);

    // this function gets called when something goes wrong
    #define DIE(...) cloverDie(__LINE__, __FILE__, __VA_ARGS__)
    void cloverDie
    (int line, const char* filename, const char* format, ...);
public:
    // kernels
    void calc_dt_kernel(double g_small, double g_big, double dtmin,
        double dtc_safe, double dtu_safe, double dtv_safe,
        double dtdiv_safe, double* dt_min_val, int* dtl_control,
        double* xl_pos, double* yl_pos, int* jldt, int* kldt, int* small);

    void field_summary_kernel(double* vol, double* mass,
        double* ie, double* ke, double* press);

    void PdV_kernel(int* error_condition, int predict, double dbyt);

    void ideal_gas_kernel(int predict);

    void generate_chunk_kernel(const int number_of_states, 
        const double* state_density, const double* state_energy,
        const double* state_xvel, const double* state_yvel,
        const double* state_xmin, const double* state_xmax,
        const double* state_ymin, const double* state_ymax,
        const double* state_radius, const int* state_geometry,
        const int g_rect, const int g_circ, const int g_point);

    void initialise_chunk_kernel(double d_xmin, double d_ymin,
        double d_dx, double d_dy);

    void update_halo_kernel(const int* fields, int depth,
        const int* chunk_neighbours);

    void accelerate_kernel(double dbyt);

    void advec_mom_kernel(int which_vel, int sweep_number, int direction);

    void flux_calc_kernel(double dbyt);

    void advec_cell_kernel(int dr, int swp_nmbr);

    void revert_kernel();

    void reset_field_kernel();

    void viscosity_kernel();

    std::map<std::string, double*> arr_names;
    std::vector<double> dumpArray
    (const std::string& arr_name, int x_extra, int y_extra);

    typedef enum {PACK, UNPACK} dir_t;
    void packRect
    (double* host_buffer, dir_t direction,
     int x_inc, int y_inc, int edge, int dest,
     int which_field, int depth);

    void pack_left_right(PACK_ARGS);
    void unpack_left_right(PACK_ARGS);
    void pack_top_bottom(PACK_ARGS);
    void unpack_top_bottom(PACK_ARGS);

    CloverleafCudaChunk
    (INITIALISE_ARGS);

    CloverleafCudaChunk
    (void);
    ~CloverleafCudaChunk
    (void);
};

extern CloverleafCudaChunk cuda_chunk;

#endif

