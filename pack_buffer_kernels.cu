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
 *  @brief CUDA mpi buffer packing
 *  @author Michael Boulton
 *  @details Packs the mpi buffers required for the mpi halo exchange
 */

#include "cuda_common.cu"

/********************/

// j is column
// k is row

// could put this check in, prob doesnt need it
// if (row > 1 - depth && row < y_max + 2 + depth + y_inc)

// left/right buffer
// index=j+(k+depth-1)*depth

// left index 
// left_snd_buffer(index)=field(chunks(chunk)%field%x_min+x_inc-1+j,k)
// field(chunks(chunk)%field%x_min-j,k)=left_rcv_buffer(index)

// right index
// right_snd_buffer(index)=field(chunks(chunk)%field%x_max+1-j,k)
// field(chunks(chunk)%field%x_max+x_inc+j,k)=right_rcv_buffer(index)

/********************/

// top/bottom buffer
// index=j+depth+(k-1)*(chunks(chunk)%field%x_max+x_inc+(2*depth))

// bottom index
// bottom_snd_buffer(index)=field(j,chunks(chunk)%field%y_min+y_inc-1+k)
// field(j,chunks(chunk)%field%y_min-k)=bottom_rcv_buffer(index)

// top index
// top_snd_buffer(index)=field(j,chunks(chunk)%field%y_max+1-k)
// field(j,chunks(chunk)%field%y_max+y_inc+k)=top_rcv_buffer(index)

/********************/

// for top/bottom
#define HORZ_IDX(add) (column + depth + ((add + 1) - 1) * (x_max + x_inc + (2 * depth)))
// for left/right
#define VERT_IDX(add) ((add + 1) + (row + depth - 1) * depth)

__global__ void device_packleftBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
const double* array,
      double* left_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int x_inc = (type.grid_type == VERTEX_DATA || type.grid_type == X_FACE_DATA) ? 1 : 0;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;

    //__kernel_indexes;
    const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = glob_id / depth;
    const int column = glob_id % depth;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_inc + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        left_buffer[VERT_IDX(column)] = array[row_begin + (x_min + 1) + x_inc - 1 + 1 + column];
    }
}

__global__ void device_unpackleftBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
      double* array,
const double* left_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;

    //__kernel_indexes;
    const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = glob_id / depth;
    const int column = glob_id % depth;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_inc + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        array[row_begin + (x_min + 1) - (1 + column)] = left_buffer[VERT_IDX(column)];
    }
}

/************************************************************/

__global__ void device_packrightBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
const double* array,
      double* right_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;

    //__kernel_indexes;
    const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = glob_id / depth;
    const int column = glob_id % depth;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_inc + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        right_buffer[VERT_IDX(column)] = array[row_begin + (x_max + 1) + 1 - (1 + column)];
    }
}

__global__ void device_unpackrightBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
      double* array,
const double* right_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int x_inc = (type.grid_type == VERTEX_DATA || type.grid_type == X_FACE_DATA) ? 1 : 0;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;

    //__kernel_indexes;
    const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = glob_id / depth;
    const int column = glob_id % depth;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_inc + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        array[row_begin + (x_max + 1) + x_inc + 1 + column] = right_buffer[VERT_IDX(column)];
    }
}

/************************************************************/

__global__ void device_packbottomBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
double* array,
double* bottom_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int x_inc = (type.grid_type == VERTEX_DATA || type.grid_type == X_FACE_DATA) ? 1 : 0;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + y_inc + depth)
    {
        if (row < depth)
        {
            bottom_buffer[HORZ_IDX(row)] = array[THARR2D(0, (y_min + 1) + y_inc - 1 + 1, x_extra)];
        }
    }
}

__global__ void device_unpackbottomBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
double* array,
double* bottom_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int x_inc = (type.grid_type == VERTEX_DATA || type.grid_type == X_FACE_DATA) ? 1 : 0;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + y_inc + depth)
    {
        if (row < depth)
        {
            array[THARR2D(0, (y_min + 1) - (1 + 2*row), x_extra)] = bottom_buffer[HORZ_IDX(row)];
        }
    }
}

/************************************************************/

__global__ void device_packtopBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
double* array,
double* top_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int x_inc = (type.grid_type == VERTEX_DATA || type.grid_type == X_FACE_DATA) ? 1 : 0;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + y_inc + depth)
    {
        if (row < depth)
        {
            top_buffer[HORZ_IDX(row)] = array[THARR2D(0, (y_max + 1) + 1 - (1 + 2*row), x_extra)];
        }
    }
}

__global__ void device_unpacktopBuffer
(int x_min,int x_max,int y_min,int y_max,
cell_info_t type,
double* array,
double* top_buffer,
const int depth)
{
    int x_extra = type.x_e;
    int x_inc = (type.grid_type == VERTEX_DATA || type.grid_type == X_FACE_DATA) ? 1 : 0;
    int y_inc = (type.grid_type == VERTEX_DATA || type.grid_type == Y_FACE_DATA) ? 1 : 0;
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + y_inc + depth)
    {
        if (row < depth)
        {
            array[THARR2D(0, (y_max + 1) + y_inc + 1, x_extra)] = top_buffer[HORZ_IDX(row)];
        }
    }
}

