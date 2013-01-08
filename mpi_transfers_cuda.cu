#include <iostream>

#include "chunk_cuda.cu"
#include "cuda_common.cu"

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
const int* x_inc,
const int* y_inc,
double* buffer,
const int* buffer_size,
const int* depth)
{
    chunk.packBuffer((*which_array) - 1, *which_side,
        *x_inc, *y_inc,
        buffer, *buffer_size, *depth);
}

// unpack from buffers
extern "C" void cudaunpackbuffers_
(const int* which_array,
const int* which_side,
const int* x_inc,
const int* y_inc,
double* buffer,
const int* buffer_size,
const int* depth)
{
    chunk.unpackBuffer((*which_array) - 1, *which_side,
        *x_inc, *y_inc,
        buffer, *buffer_size, *depth);
}

void CloverleafCudaChunk::packBuffer
(const int which_array,
const int which_side,
const int x_inc,
const int y_inc,
double* buffer,
const int buffer_size,
const int depth)
{
    #define PACK_CUDA_BUFFERS(dev_ptr, x_extra, y_extra) \
        switch(which_side) \
        { \
            case LEFT_FACE: \
                device_packLeftBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_left_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(buffer, dev_left_buffer, buffer_size*sizeof(double), cudaMemcpyDeviceToHost); \
                errChk(__LINE__, __FILE__); \
                break;\
            case RIGHT_FACE:\
                device_packRightBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_right_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(buffer, dev_right_buffer, buffer_size*sizeof(double), cudaMemcpyDeviceToHost); \
                errChk(__LINE__, __FILE__); \
                break;\
            case TOP_FACE:\
                device_packTopBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_top_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(buffer, dev_top_buffer, buffer_size*sizeof(double), cudaMemcpyDeviceToHost); \
                errChk(__LINE__, __FILE__); \
                break;\
            case BOTTOM_FACE:\
                device_packBottomBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_bottom_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(buffer, dev_bottom_buffer, buffer_size*sizeof(double), cudaMemcpyDeviceToHost); \
                errChk(__LINE__, __FILE__); \
                break;\
            default: \
                std::cout << "Invalid side passed to buffer packing in task " << task << std::endl; \
                exit(1); \
        }

    switch(which_array)
    {
        case FIELD_density0: PACK_CUDA_BUFFERS(density0, 0, 0); break;
        case FIELD_density1: PACK_CUDA_BUFFERS(density1, 0, 0); break;
        case FIELD_energy0: PACK_CUDA_BUFFERS(energy0, 0, 0); break;
        case FIELD_energy1: PACK_CUDA_BUFFERS(energy1, 0, 0); break;
        case FIELD_pressure: PACK_CUDA_BUFFERS(pressure, 0, 0); break;
        case FIELD_viscosity: PACK_CUDA_BUFFERS(viscosity, 0, 0); break;
        case FIELD_soundspeed: PACK_CUDA_BUFFERS(soundspeed, 0, 0); break;
        case FIELD_xvel0: PACK_CUDA_BUFFERS(xvel0, 1, 1); break;
        case FIELD_xvel1: PACK_CUDA_BUFFERS(xvel1, 1, 1); break;
        case FIELD_yvel0: PACK_CUDA_BUFFERS(yvel0, 1, 1); break;
        case FIELD_yvel1: PACK_CUDA_BUFFERS(yvel1, 1, 1); break;
        case FIELD_vol_flux_x: PACK_CUDA_BUFFERS(vol_flux_x, 1, 0); break;
        case FIELD_vol_flux_y: PACK_CUDA_BUFFERS(vol_flux_y, 0, 1); break;
        case FIELD_mass_flux_x: PACK_CUDA_BUFFERS(mass_flux_x, 1, 0); break;
        case FIELD_mass_flux_y: PACK_CUDA_BUFFERS(mass_flux_y, 0, 1); break;
        default: std::cerr << "Invalid which_array identifier passed to CUDA for MPI transfer" << std::endl; exit(1);
    }
}

void CloverleafCudaChunk::unpackBuffer
(const int which_array,
const int which_side,
const int x_inc,
const int y_inc,
double* buffer,
const int buffer_size,
const int depth)
{
    #define UNPACK_CUDA_BUFFERS(dev_ptr, x_extra, y_extra) \
        switch(which_side) \
        { \
            case LEFT_FACE: \
                device_unpackLeftBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_left_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(dev_left_buffer, buffer, buffer_size*sizeof(double), cudaMemcpyHostToDevice); \
                errChk(__LINE__, __FILE__); \
                break;\
            case RIGHT_FACE:\
                device_unpackRightBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_right_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(dev_right_buffer, buffer, buffer_size*sizeof(double), cudaMemcpyHostToDevice); \
                errChk(__LINE__, __FILE__); \
                break;\
            case TOP_FACE:\
                device_unpackTopBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_top_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(dev_top_buffer, buffer, buffer_size*sizeof(double), cudaMemcpyHostToDevice); \
                errChk(__LINE__, __FILE__); \
                break;\
            case BOTTOM_FACE:\
                device_unpackBottomBuffer<<< num_blocks, BLOCK_SZ >>> \
                (x_min, x_max, y_min, y_max,\
                x_extra, y_extra, \
                x_inc, y_inc, \
                dev_ptr, dev_bottom_buffer, buffer_size, depth); \
                errChk(__LINE__, __FILE__); \
                cudaMemcpy(dev_bottom_buffer, buffer, buffer_size*sizeof(double), cudaMemcpyHostToDevice); \
                errChk(__LINE__, __FILE__); \
                break;\
            default: \
                std::cout << "Invalid side passed to buffer unpacking in task " << task << std::endl; \
                exit(1); \
        }

    switch(which_array)
    {
        case FIELD_density0: UNPACK_CUDA_BUFFERS(density0, 0, 0); break;
        case FIELD_density1: UNPACK_CUDA_BUFFERS(density1, 0, 0); break;
        case FIELD_energy0: UNPACK_CUDA_BUFFERS(energy0, 0, 0); break;
        case FIELD_energy1: UNPACK_CUDA_BUFFERS(energy1, 0, 0); break;
        case FIELD_pressure: UNPACK_CUDA_BUFFERS(pressure, 0, 0); break;
        case FIELD_viscosity: UNPACK_CUDA_BUFFERS(viscosity, 0, 0); break;
        case FIELD_soundspeed: UNPACK_CUDA_BUFFERS(soundspeed, 0, 0); break;
        case FIELD_xvel0: UNPACK_CUDA_BUFFERS(xvel0, 1, 1); break;
        case FIELD_xvel1: UNPACK_CUDA_BUFFERS(xvel1, 1, 1); break;
        case FIELD_yvel0: UNPACK_CUDA_BUFFERS(yvel0, 1, 1); break;
        case FIELD_yvel1: UNPACK_CUDA_BUFFERS(yvel1, 1, 1); break;
        case FIELD_vol_flux_x: UNPACK_CUDA_BUFFERS(vol_flux_x, 1, 0); break;
        case FIELD_vol_flux_y: UNPACK_CUDA_BUFFERS(vol_flux_y, 0, 1); break;
        case FIELD_mass_flux_x: UNPACK_CUDA_BUFFERS(mass_flux_x, 1, 0); break;
        case FIELD_mass_flux_y: UNPACK_CUDA_BUFFERS(mass_flux_y, 0, 1); break;
        default: std::cerr << "Invalid which_array identifier passed to CUDA for MPI transfer" << std::endl; exit(1);
    }
}

/****************************************************/

// copy data out from device back to host
extern "C" void clover_cuda_copy_out_
(const int* which_array, double* copy_into)
{
    chunk.copyToHost((*which_array) - 1, copy_into);
}

// copy data back to device from host
extern "C" void clover_cuda_copy_back_
(const int* which_array, double* copy_from)
{
    chunk.copyToDevice((*which_array) - 1, copy_from);
}
void CloverleafCudaChunk::copyToHost(const int which_array, double* copy_into)
{
    #define COPY_OUT(dev_ptr, sz) \
        std::cout << "copying to host: " << #dev_ptr << std::endl; \
        cudaMemcpy(copy_into, dev_ptr, sz, cudaMemcpyDeviceToHost); \
        errChk(__LINE__, __FILE__);
    switch(which_array)
    {
        case FIELD_density0: COPY_OUT(density0, BUFSZ2D(0, 0)); break;
        case FIELD_density1: COPY_OUT(density1, BUFSZ2D(0, 0)); break;
        case FIELD_energy0: COPY_OUT(energy0, BUFSZ2D(0, 0)); break;
        case FIELD_energy1: COPY_OUT(energy1, BUFSZ2D(0, 0)); break;
        case FIELD_pressure: COPY_OUT(pressure, BUFSZ2D(0, 0)); break;
        case FIELD_viscosity: COPY_OUT(viscosity, BUFSZ2D(0, 0)); break;
        case FIELD_soundspeed: COPY_OUT(soundspeed, BUFSZ2D(0, 0)); break;
        case FIELD_xvel0: COPY_OUT(xvel0, BUFSZ2D(1, 1)); break;
        case FIELD_xvel1: COPY_OUT(xvel1, BUFSZ2D(1, 1)); break;
        case FIELD_yvel0: COPY_OUT(yvel0, BUFSZ2D(1, 1)); break;
        case FIELD_yvel1: COPY_OUT(yvel1, BUFSZ2D(1, 1)); break;
        case FIELD_vol_flux_x: COPY_OUT(vol_flux_x, BUFSZ2D(1, 0)); break;
        case FIELD_vol_flux_y: COPY_OUT(vol_flux_y, BUFSZ2D(0, 1)); break;
        case FIELD_mass_flux_x: COPY_OUT(mass_flux_x, BUFSZ2D(1, 0)); break;
        case FIELD_mass_flux_y: COPY_OUT(mass_flux_y, BUFSZ2D(0, 1)); break;
        default: std::cerr << "Invalid which_array identifier passed to CUDA for MPI transfer" << std::endl; exit(1);
    }
}

void CloverleafCudaChunk::copyToDevice(const int which_array, double* copy_from)
{
    #define COPY_BACK(dev_ptr, sz) \
        std::cout << "copying to device: " << #dev_ptr << std::endl; \
        cudaMemcpy(dev_ptr, copy_from, sz, cudaMemcpyHostToDevice); \
        errChk(__LINE__, __FILE__);
    switch(which_array)
    {
        case FIELD_density0: COPY_BACK(density0, BUFSZ2D(0, 0)); break;
        case FIELD_density1: COPY_BACK(density1, BUFSZ2D(0, 0)); break;
        case FIELD_energy0: COPY_BACK(energy0, BUFSZ2D(0, 0)); break;
        case FIELD_energy1: COPY_BACK(energy1, BUFSZ2D(0, 0)); break;
        case FIELD_pressure: COPY_BACK(pressure, BUFSZ2D(0, 0)); break;
        case FIELD_viscosity: COPY_BACK(viscosity, BUFSZ2D(0, 0)); break;
        case FIELD_soundspeed: COPY_BACK(soundspeed, BUFSZ2D(0, 0)); break;
        case FIELD_xvel0: COPY_BACK(xvel0, BUFSZ2D(1, 1)); break;
        case FIELD_xvel1: COPY_BACK(xvel1, BUFSZ2D(1, 1)); break;
        case FIELD_yvel0: COPY_BACK(yvel0, BUFSZ2D(1, 1)); break;
        case FIELD_yvel1: COPY_BACK(yvel1, BUFSZ2D(1, 1)); break;
        case FIELD_vol_flux_x: COPY_BACK(vol_flux_x, BUFSZ2D(1, 0)); break;
        case FIELD_vol_flux_y: COPY_BACK(vol_flux_y, BUFSZ2D(0, 1)); break;
        case FIELD_mass_flux_x: COPY_BACK(mass_flux_x, BUFSZ2D(1, 0)); break;
        case FIELD_mass_flux_y: COPY_BACK(mass_flux_y, BUFSZ2D(0, 1)); break;
        default: std::cerr << "Invalid which_array identifier passed to CUDA for MPI transfer" << std::endl; exit(1);
    }
}

