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
 *  @brief CUDA initialisation
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Initialises CUDA devices and global storage
 */

#if defined(MPI_HDR)
extern "C" void clover_get_rank_(int*);
#endif

#include "cuda_common.hpp"
#include "cuda_strings.hpp"

#include <sstream>
#include <cstdio>
#include <cassert>

CloverleafCudaChunk cuda_chunk;

extern "C" void initialise_cuda_
(INITIALISE_ARGS)
{
    cuda_chunk = CloverleafCudaChunk(in_x_min,
                                in_x_max,
                                in_y_min,
                                in_y_max,
                                in_profiler_on);
}

CloverleafCudaChunk::CloverleafCudaChunk
(void)
{
    ;
}

CloverleafCudaChunk::CloverleafCudaChunk
(INITIALISE_ARGS)
:x_min(*in_x_min),
x_max(*in_x_max),
y_min(*in_y_min),
y_max(*in_y_max),
profiler_on(*in_profiler_on),
num_blocks((((*in_x_max)+5)*((*in_y_max)+5))/BLOCK_SZ)
{
    // FIXME (and opencl really)
    // make a better platform agnostic way of selecting devices

    int rank;
#if defined(MPI_HDR)
    clover_get_rank_(&rank);
#else
    rank = 0;
#endif

    // Read in from file - easier than passing in from fortran
    FILE* input = fopen("clover.in", "r");
    if (NULL == input)
    {
        // should never happen
        DIE("Input file not found\n");
    }

    int device_id = clover::preferredDevice(input);
    device_id = device_id < 0 ? 0 : device_id;

    fclose(input);

#ifdef MANUALLY_CHOOSE_GPU
    // choose device 0 unless specified
    cudaThreadExit();
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    fprintf(stdout, "%d devices available in rank %d - would use %d - adding %d - choosing %d\n",
            num_devices, rank, device_id, rank%num_devices, device_id + rank % num_devices);
    fflush(stdout);
    device_id += rank % num_devices;

    int err = cudaSetDevice(device_id);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Setting device id to %d in rank %d failed with error code %d\n", device_id, rank, err);
        errorHandler(__LINE__, __FILE__);
    }
#endif

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "CUDA using " << prop.name << std::endl;

    #define CUDA_ARRAY_ALLOC(arr, size)     \
            cudaMalloc((void**) &arr, size) == cudaSuccess;\
            errorHandler(__LINE__, __FILE__);\
            cudaDeviceSynchronize();        \
            cudaMemset(arr, 0, size);       \
            cudaDeviceSynchronize();        \
            CUDA_ERR_CHECK;

    CUDA_ARRAY_ALLOC(volume, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(soundspeed, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(pressure, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(viscosity, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(density0, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(density1, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(energy0, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(energy1, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(xvel0, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(xvel1, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(yvel0, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(yvel1, BUFSZ2D(1, 1));

    CUDA_ARRAY_ALLOC(xarea, BUFSZ2D(1, 0));
    CUDA_ARRAY_ALLOC(vol_flux_x, BUFSZ2D(1, 0));
    CUDA_ARRAY_ALLOC(mass_flux_x, BUFSZ2D(1, 0));

    CUDA_ARRAY_ALLOC(yarea, BUFSZ2D(0, 1));
    CUDA_ARRAY_ALLOC(vol_flux_y, BUFSZ2D(0, 1));
    CUDA_ARRAY_ALLOC(mass_flux_y, BUFSZ2D(0, 1));

    CUDA_ARRAY_ALLOC(cellx, BUFSZX(0));
    CUDA_ARRAY_ALLOC(celldx, BUFSZX(0));
    CUDA_ARRAY_ALLOC(vertexx, BUFSZX(1));
    CUDA_ARRAY_ALLOC(vertexdx, BUFSZX(1));

    CUDA_ARRAY_ALLOC(celly, BUFSZY(0));
    CUDA_ARRAY_ALLOC(celldy, BUFSZY(0));
    CUDA_ARRAY_ALLOC(vertexy, BUFSZY(1));
    CUDA_ARRAY_ALLOC(vertexdy, BUFSZY(1));

    CUDA_ARRAY_ALLOC(work_array_1, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_2, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_3, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_4, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_5, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_6, BUFSZ2D(1, 1));

    CUDA_ARRAY_ALLOC(reduce_buf_1, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_2, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_3, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_4, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_5, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_6, num_blocks*sizeof(double));

    reduce_ptr_1 = thrust::device_ptr< double >(reduce_buf_1);
    reduce_ptr_2 = thrust::device_ptr< double >(reduce_buf_2);
    reduce_ptr_3 = thrust::device_ptr< double >(reduce_buf_3);
    reduce_ptr_4 = thrust::device_ptr< double >(reduce_buf_4);
    reduce_ptr_5 = thrust::device_ptr< double >(reduce_buf_5);
    reduce_ptr_6 = thrust::device_ptr< double >(reduce_buf_6);

    CUDA_ARRAY_ALLOC(pdv_reduce_array, num_blocks*sizeof(int));
    reduce_pdv = thrust::device_ptr< int >(pdv_reduce_array);

    thr_cellx = thrust::device_ptr< double >(cellx);
    thr_celly = thrust::device_ptr< double >(celly);
    thr_xvel0 = thrust::device_ptr< double >(xvel0);
    thr_yvel0 = thrust::device_ptr< double >(yvel0);
    thr_xvel1 = thrust::device_ptr< double >(xvel1);
    thr_yvel1 = thrust::device_ptr< double >(yvel1);
    thr_density0 = thrust::device_ptr< double >(density0);
    thr_energy0 = thrust::device_ptr< double >(energy0);
    thr_pressure = thrust::device_ptr< double >(pressure);
    thr_soundspeed = thrust::device_ptr< double >(soundspeed);

    CUDA_ARRAY_ALLOC(dev_left_send_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_right_send_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_top_send_buffer, sizeof(double)*(x_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_bottom_send_buffer, sizeof(double)*(x_max+5)*2);

    CUDA_ARRAY_ALLOC(dev_left_recv_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_right_recv_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_top_recv_buffer, sizeof(double)*(x_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_bottom_recv_buffer, sizeof(double)*(x_max+5)*2);

    #undef CUDA_ARRAY_ALLOC

#define ADD_BUFFER_DBG_MAP(name) arr_names[#name] = name;
    ADD_BUFFER_DBG_MAP(volume);
    ADD_BUFFER_DBG_MAP(soundspeed);
    ADD_BUFFER_DBG_MAP(pressure);
    ADD_BUFFER_DBG_MAP(viscosity);

    ADD_BUFFER_DBG_MAP(work_array_2);
    ADD_BUFFER_DBG_MAP(work_array_3);
    ADD_BUFFER_DBG_MAP(work_array_4);
    ADD_BUFFER_DBG_MAP(work_array_5);
    ADD_BUFFER_DBG_MAP(work_array_6);

    ADD_BUFFER_DBG_MAP(density0);
    ADD_BUFFER_DBG_MAP(density1);
    ADD_BUFFER_DBG_MAP(energy0);
    ADD_BUFFER_DBG_MAP(energy1);
    ADD_BUFFER_DBG_MAP(xvel0);
    ADD_BUFFER_DBG_MAP(xvel1);
    ADD_BUFFER_DBG_MAP(yvel0);
    ADD_BUFFER_DBG_MAP(yvel1);
    ADD_BUFFER_DBG_MAP(xarea);
    ADD_BUFFER_DBG_MAP(yarea);
    ADD_BUFFER_DBG_MAP(vol_flux_x);
    ADD_BUFFER_DBG_MAP(vol_flux_y);
    ADD_BUFFER_DBG_MAP(mass_flux_x);
    ADD_BUFFER_DBG_MAP(mass_flux_y);

    ADD_BUFFER_DBG_MAP(cellx);
    ADD_BUFFER_DBG_MAP(celly);
    ADD_BUFFER_DBG_MAP(celldx);
    ADD_BUFFER_DBG_MAP(celldy);
    ADD_BUFFER_DBG_MAP(vertexx);
    ADD_BUFFER_DBG_MAP(vertexy);
    ADD_BUFFER_DBG_MAP(vertexdx);
    ADD_BUFFER_DBG_MAP(vertexdy);
#undef ADD_BUFFER_DBG_MAP
}

