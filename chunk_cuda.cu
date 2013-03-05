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
 *  @brief CUDA Chunk definition
 *  @author Michael Boulton
 *  @details class definition for a cuda implementation of the 'chunk' in the original fortran.
 */

#define INITIALISE_ARGS \
    /* values used to control operation */\
    int* in_x_min, \
    int* in_x_max, \
    int* in_y_min, \
    int* in_y_max, \
 \
    int* in_left, \
    int* in_right, \
    int* in_top, \
    int* in_bottom, \
    int* in_left_boundary, \
    int* in_right_boundary, \
    int* in_top_boundary, \
    int* in_bottom_boundary, \
 \
    /* mpi task */\
    int* in_task \

#include "thrust/copy.h"

#include "cuda_common.cu"

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

    // buffers used in mpi transfers
    double* dev_left_send_buffer;
    double* dev_right_send_buffer;
    double* dev_top_send_buffer;
    double* dev_bottom_send_buffer;
    double* dev_left_recv_buffer;
    double* dev_right_recv_buffer;
    double* dev_top_recv_buffer;
    double* dev_bottom_recv_buffer;

    // used for reductions in calc dt, pdv, field summary
    thrust::device_ptr< double > reduce_ptr_1;
    thrust::device_ptr< double > reduce_ptr_2;
    thrust::device_ptr< double > reduce_ptr_3;
    thrust::device_ptr< double > reduce_ptr_4;
    thrust::device_ptr< double > reduce_ptr_5;

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

    int left;
    int right;
    int top;
    int bottom;
    int left_boundary;
    int right_boundary;
    int top_boundary;
    int bottom_boundary;

    // mpi task
    int task;
public:
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
        const int g_rect, const int g_circ);

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

    // mpi functions
    void copyToHost(const int array, double* copy_from);
    void copyToDevice(const int array, double* copy_into);

    void packBuffer (const int which_array, const int which_side,
        double* buffer, const int buffer_size, const int depth);
    void unpackBuffer (const int which_array, const int which_side,
        double* buffer, const int buffer_size, const int depth);

    CloverleafCudaChunk
    (INITIALISE_ARGS);

    CloverleafCudaChunk
    (void)
    {
        ;
    }

};

