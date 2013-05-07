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
#include "chunk_cuda.cu"
#include "cuda_common.cu"

#include <numeric>

#include "pack_buffer_kernels.cu"

extern CloverleafCudaChunk chunk;

// which side to pack - keep the same as in fortran file
#define LEFT_FACE 0
#define RIGHT_FACE 1
#define TOP_FACE 2
#define BOTTOM_FACE 3

/**********************/

// pack data into buffers
extern "C" void cudapackbuffers_
(const int* which_array,
const int* which_side,
double* buffer,
const int* buffer_size,
const int* depth)
{
    chunk.packBuffer((*which_array) - 1, *which_side,
        buffer, *buffer_size, *depth);
}

// unpack from buffers
extern "C" void cudaunpackbuffers_
(const int* which_array,
const int* which_side,
double* buffer,
const int* buffer_size,
const int* depth)
{
    chunk.unpackBuffer((*which_array) - 1, *which_side,
        buffer, *buffer_size, *depth);
}

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
            case LEFT_FACE: \
                CALL_PACK(dev_ptr, type, left, y);\
            case RIGHT_FACE:\
                CALL_PACK(dev_ptr, type, right, y);\
            case BOTTOM_FACE:\
                CALL_PACK(dev_ptr, type, bottom, x);\
            case TOP_FACE:\
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
            case LEFT_FACE: \
                CALL_UNPACK(dev_ptr, type, left, y);\
            case RIGHT_FACE:\
                CALL_UNPACK(dev_ptr, type, right, y);\
            case BOTTOM_FACE:\
                CALL_UNPACK(dev_ptr, type, bottom, x);\
            case TOP_FACE:\
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

