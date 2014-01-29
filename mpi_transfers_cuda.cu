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
 *  @brief CUDA mpi buffer transfer
 *  @author Michael Boulton
 *  @details Transfers the buffers required for the mpi halo exchange
 */

#include <iostream>
#include "cuda_common.cu"
#include "chunk_cuda.cu"

#include <numeric>

#include "pack_buffer_kernels.cu"

#define CHUNK_LEFT 1
#define CHUNK_RIGHT 2
#define CHUNK_BOTTOM 3
#define CHUNK_TOP 4
#define EXTERNAL_FACE (-1)

extern CloverleafCudaChunk chunk;

/**********************/

// define a generic interface for fortran
#define C_PACK_INTERFACE(operation, dir)                            \
extern "C" void operation##_##dir##_buffers_cuda_                    \
(int *xmin, int *xmax, int *ymin, int *ymax,                        \
 int *chunk_1, int *chunk_2, int *external_face,                    \
 int *x_inc, int *y_inc, int *depth, int *which_field,              \
 double *field_ptr, double *buffer_1, double *buffer_2)             \
{                                                                   \
    chunk.operation##_##dir(*chunk_1, *chunk_2, *external_face,     \
                            *x_inc, *y_inc, *depth,                 \
                            (*which_field)-1, buffer_1, buffer_2);  \
}

C_PACK_INTERFACE(pack, left_right)
C_PACK_INTERFACE(unpack, left_right)
C_PACK_INTERFACE(pack, top_bottom)
C_PACK_INTERFACE(unpack, top_bottom)

/*****************************/

void CloverleafCudaChunk::packBuffer
(const int which_array,
const int which_side,
double* buffer,
const int buffer_size,
const int depth)
{
    #define CALL_PACK(dev_ptr, type, face, dir)\
	{\
        const int launch_sz = (ceil((dir##_max+4+type.dir##_e)/static_cast<float>(BLOCK_SZ))) * depth; \
        device_pack##face##Buffer<<< launch_sz, BLOCK_SZ >>> \
        (x_min, x_max, y_min, y_max, type, \
        dev_ptr, dev_##face##_send_buffer, depth); \
        CUDA_ERR_CHECK; \
        cudaMemcpy(buffer, dev_##face##_send_buffer, buffer_size*sizeof(double), cudaMemcpyDeviceToHost); \
        CUDA_ERR_CHECK; \
        cudaDeviceSynchronize();\
        break; \
	}

    #define PACK_CUDA_BUFFERS(dev_ptr, type) \
        switch(which_side) \
        { \
            case CHUNK_LEFT: \
                CALL_PACK(dev_ptr, type, left, y);\
            case CHUNK_RIGHT:\
                CALL_PACK(dev_ptr, type, right, y);\
            case CHUNK_BOTTOM:\
                CALL_PACK(dev_ptr, type, bottom, x);\
            case CHUNK_TOP:\
                CALL_PACK(dev_ptr, type, top, x);\
            default: \
                std::cout << "Invalid side passed to buffer packing in " << __FILE__ << std::endl; \
                exit(1); \
        }

    switch(which_array)
    {
        case FIELD_density0: PACK_CUDA_BUFFERS(density0, CELL); break;
        case FIELD_density1: PACK_CUDA_BUFFERS(density1, CELL); break;
        case FIELD_energy0: PACK_CUDA_BUFFERS(energy0, CELL); break;
        case FIELD_energy1: PACK_CUDA_BUFFERS(energy1, CELL); break;
        case FIELD_pressure: PACK_CUDA_BUFFERS(pressure, CELL); break;
        case FIELD_viscosity: PACK_CUDA_BUFFERS(viscosity, CELL); break;
        case FIELD_soundspeed: PACK_CUDA_BUFFERS(soundspeed, CELL); break;
        case FIELD_xvel0: PACK_CUDA_BUFFERS(xvel0, VERTEX_X); break;
        case FIELD_xvel1: PACK_CUDA_BUFFERS(xvel1, VERTEX_X); break;
        case FIELD_yvel0: PACK_CUDA_BUFFERS(yvel0, VERTEX_Y); break;
        case FIELD_yvel1: PACK_CUDA_BUFFERS(yvel1, VERTEX_Y); break;
        case FIELD_vol_flux_x: PACK_CUDA_BUFFERS(vol_flux_x, X_FACE); break;
        case FIELD_vol_flux_y: PACK_CUDA_BUFFERS(vol_flux_y, Y_FACE); break;
        case FIELD_mass_flux_x: PACK_CUDA_BUFFERS(mass_flux_x, X_FACE); break;
        case FIELD_mass_flux_y: PACK_CUDA_BUFFERS(mass_flux_y, Y_FACE); break;
        default: std::cerr << "Invalid which_array identifier passed to CUDA for MPI transfer" << std::endl; exit(1);
    }

}

void CloverleafCudaChunk::unpackBuffer
(const int which_array,
const int which_side,
double* buffer,
const int buffer_size,
const int depth)
{
    #define CALL_UNPACK(dev_ptr, type, face, dir)\
	{ \
        cudaMemcpy(dev_##face##_recv_buffer, buffer, buffer_size*sizeof(double), cudaMemcpyHostToDevice); \
        CUDA_ERR_CHECK; \
        cudaDeviceSynchronize();\
        const int launch_sz = (ceil((dir##_max+4+type.dir##_e)/static_cast<float>(BLOCK_SZ))) * depth; \
        device_unpack##face##Buffer<<< launch_sz, BLOCK_SZ >>> \
        (x_min, x_max, y_min, y_max, type, \
        dev_ptr, dev_##face##_recv_buffer, depth); \
        CUDA_ERR_CHECK; \
        break; \
	}

    #define UNPACK_CUDA_BUFFERS(dev_ptr, type) \
        switch(which_side) \
        { \
            case CHUNK_LEFT: \
                CALL_UNPACK(dev_ptr, type, left, y);\
            case CHUNK_RIGHT:\
                CALL_UNPACK(dev_ptr, type, right, y);\
            case CHUNK_BOTTOM:\
                CALL_UNPACK(dev_ptr, type, bottom, x);\
            case CHUNK_TOP:\
                CALL_UNPACK(dev_ptr, type, top, x);\
            default: \
                std::cout << "Invalid side passed to buffer packing in " << __FILE__ << std::endl; \
                exit(1); \
        }

    switch(which_array)
    {
        case FIELD_density0: UNPACK_CUDA_BUFFERS(density0, CELL); break;
        case FIELD_density1: UNPACK_CUDA_BUFFERS(density1, CELL); break;
        case FIELD_energy0: UNPACK_CUDA_BUFFERS(energy0, CELL); break;
        case FIELD_energy1: UNPACK_CUDA_BUFFERS(energy1, CELL); break;
        case FIELD_pressure: UNPACK_CUDA_BUFFERS(pressure, CELL); break;
        case FIELD_viscosity: UNPACK_CUDA_BUFFERS(viscosity, CELL); break;
        case FIELD_soundspeed: UNPACK_CUDA_BUFFERS(soundspeed, CELL); break;
        case FIELD_xvel0: UNPACK_CUDA_BUFFERS(xvel0, VERTEX_X); break;
        case FIELD_xvel1: UNPACK_CUDA_BUFFERS(xvel1, VERTEX_X); break;
        case FIELD_yvel0: UNPACK_CUDA_BUFFERS(yvel0, VERTEX_Y); break;
        case FIELD_yvel1: UNPACK_CUDA_BUFFERS(yvel1, VERTEX_Y); break;
        case FIELD_vol_flux_x: UNPACK_CUDA_BUFFERS(vol_flux_x, X_FACE); break;
        case FIELD_vol_flux_y: UNPACK_CUDA_BUFFERS(vol_flux_y, Y_FACE); break;
        case FIELD_mass_flux_x: UNPACK_CUDA_BUFFERS(mass_flux_x, X_FACE); break;
        case FIELD_mass_flux_y: UNPACK_CUDA_BUFFERS(mass_flux_y, Y_FACE); break;
        default: std::cerr << "Invalid which_array identifier passed to CUDA for MPI transfer" << std::endl; exit(1);
    }

}

int CloverleafCudaChunk::getBufferSize
(int edge, int depth, int x_inc, int y_inc)
{
    int region[2];

    switch (edge)
    {
    // depth*y_max+... region - 1 or 2 columns
    case CHUNK_LEFT:
        region[0] = depth;
        region[1] = y_max + y_inc + (2*depth);
        break;
    case CHUNK_RIGHT:
        region[0] = depth;
        region[1] = y_max + y_inc + (2*depth);
        break;

    // depth*x_max+... region - 1 or 2 rows
    case CHUNK_BOTTOM:
        region[0] = x_max + x_inc + (2*depth);
        region[1] = depth;
        break;
    case CHUNK_TOP:
        region[0] = x_max + x_inc + (2*depth);
        region[1] = depth;
        break;
    default:
        std::cerr << "Invalid face identifier " << edge << " passed to left/right pack buffer" << std::endl;
        exit(1);
    }

    return region[0]*region[1];
}

#define CHECK_PACK(op, side1, side2)                          \
    if (external_face != chunk_1 || external_face != chunk_2)               \
    {                                                                       \
        cudaDeviceSynchronize();                                            \
    } \
    if (external_face != chunk_1)                                           \
    {                                                                       \
        op##Buffer(which_field, \
                   chunk_1, \
                   buffer_1, \
                   getBufferSize(chunk_1, depth, x_inc, y_inc), \
                   depth); \
    }                                                                       \
    if (external_face != chunk_2)                                           \
    {                                                                       \
        op##Buffer(which_field, \
                   chunk_2, \
                   buffer_2, \
                   getBufferSize(chunk_2, depth, x_inc, y_inc), \
                   depth); \
    }                                                                       \
    if (external_face != chunk_1 || external_face != chunk_2)               \
    {                                                                       \
        cudaDeviceSynchronize();                                            \
    }

void CloverleafCudaChunk::pack_left_right
(PACK_ARGS)
{
    CHECK_PACK(pack, CHUNK_LEFT, CHUNK_RIGHT);
}

void CloverleafCudaChunk::unpack_left_right
(PACK_ARGS)
{
    CHECK_PACK(unpack, CHUNK_LEFT, CHUNK_RIGHT);
}

void CloverleafCudaChunk::pack_top_bottom
(PACK_ARGS)
{
    CHECK_PACK(pack, CHUNK_BOTTOM, CHUNK_TOP);
}

void CloverleafCudaChunk::unpack_top_bottom
(PACK_ARGS)
{
    CHECK_PACK(unpack, CHUNK_BOTTOM, CHUNK_TOP);
}

