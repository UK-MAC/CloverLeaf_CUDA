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

// size of workgroup/block - 256 seems to be optimal
#ifndef BLOCK_SZ 
    #define BLOCK_SZ 256
#endif

// number of bytes to allocate for x size array
#define BUFSZX(x_extra)   \
    ( ((x_max) + 4 + x_extra)       \
    * sizeof(double) )

// number of bytes to allocate for y size array
#define BUFSZY(y_extra)   \
    ( ((y_max) + 4 + y_extra)       \
    * sizeof(double) )

// number of bytes to allocate for 2d array
#define BUFSZ2D(x_extra, y_extra)   \
    ( ((x_max) + 4 + x_extra)       \
    * ((y_max) + 4 + y_extra)       \
    * sizeof(double) )

/*
*  access a value in a 2d array given the x and y offset from current thread
*  index, adding or subtracting a bit more if it is one of the arrays with
*  bigger rows
*/
#define THARR2D(x_offset, y_offset, big_row)\
    ( glob_id                               \
    + (x_offset)                            \
    + ((y_offset) * (x_max + 4))            \
    + (big_row * (row + (y_offset))) )

// kernel indexes uses in all kernels
#define __kernel_indexes                    \
    const int glob_id = threadIdx.x         \
        + blockIdx.x * blockDim.x;          \
    const int row = glob_id / (x_max + 4);  \
    const int column = glob_id % (x_max + 4);

// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density0      0
#define FIELD_density1      1
#define FIELD_energy0       2
#define FIELD_energy1       3
#define FIELD_pressure      4
#define FIELD_viscosity     5
#define FIELD_soundspeed    6
#define FIELD_xvel0         7
#define FIELD_xvel1         8
#define FIELD_yvel0         9
#define FIELD_yvel1         10
#define FIELD_vol_flux_x    11
#define FIELD_vol_flux_y    12
#define FIELD_mass_flux_x   13
#define FIELD_mass_flux_y   14
#define NUM_FIELDS          15

enum {CELL_DATA, VERTEX_DATA, X_FACE_DATA, Y_FACE_DATA};

/*******************/

// disable checking for errors after kernel calls / memory allocation
#ifdef NO_ERR_CHK

// do nothing instead
#define CUDA_ERR_CHECK ;

#else

#include <iostream>

#define CUDA_ERR_CHECK errorHandler(__LINE__, __FILE__);

static const char* errorCodes
(int err_code)
{
    switch(err_code)
    {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
        case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
        case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
        case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
        case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
        case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
        case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
        case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
        case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
        case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
        case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
        case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
        case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
        case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
        case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
        case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
        case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
        case cudaErrorUnknown: return "cudaErrorUnknown";
        case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
        case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
        case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
        case cudaErrorNotReady: return "cudaErrorNotReady";
        case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";
        case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";
        case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";
        case cudaErrorNoDevice: return "cudaErrorNoDevice";
        case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";
        case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";
        case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";
        case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";
        case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";
        case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";
        case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";
        case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";
        case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";
        case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
        case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";
        case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";
        case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";
        case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse";
        case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled";
        case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized";
        case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted";
        case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped";
        case cudaErrorAssert: return "cudaErrorAssert";
        case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";
        case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";
        case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";
        case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";
        case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
        case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
        default: return "Unknown error";
    }
}

inline void errorHandler
(int line_num, std::string const& file)
{
    cudaDeviceSynchronize();
    int l_e = cudaGetLastError();
    if (cudaSuccess != l_e)
    {
        std::cout << "error on line " << line_num << " of ";
        std::cout << file << std::endl;
        std::cout << "return code " << l_e; 
        std::cout << " (" << errorCodes(l_e) << ")";
        std::cout << std::endl;
        exit(l_e);
    }
}

#endif //NO_ERR_CHK

// whether to time kernel run times
#ifdef TIME_KERNELS

// beginning of profiling bit
#define CUDA_BEGIN_PROFILE \
    double __t_0, __t_1;          \
    __t_0 = MPI_Wtime();

// end of profiling bit
#define CUDA_END_PROFILE \
    cudaDeviceSynchronize();                        \
    __t_1 = MPI_Wtime();                       \
    std::cout << "[PROFILING] " << __t_1 - __t_0  \
    << " to calculate " << __FILE__  << std::endl;

#else

#define CUDA_BEGIN_PROFILE ;
#define CUDA_END_PROFILE ;

#endif // TIME_KERNELS

typedef struct cell_info {
    const int x_e;
    const int y_e;
    const int x_i;
    const int y_i;
    const int x_f;
    const int y_f;
    const int grid_type;

    cell_info
    (int x_extra, int y_extra,
    int x_invert, int y_invert,
    int x_face, int y_face,
    int in_type)
    :x_e(x_extra), y_e(y_extra),
    x_i(x_invert), y_i(y_invert),
    x_f(x_face), y_f(y_face),
    grid_type(in_type)
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

#include "ftocmacros.h"

// callbacks for reductions
__device__ inline static int sum_func (int x, int y) { return x + y; }
__device__ inline static int min_func (int x, int y) { return MIN(x, y); }
__device__ inline static int max_func (int x, int y) { return MAX(x, y); }

__device__ inline static double sum_func (double x, double y) { return x + y; }
__device__ inline static double min_func (double x, double y) { return MIN(x, y); }
__device__ inline static double max_func (double x, double y) { return MAX(x, y); }

template < int offset >
class Reduce
{
public:
    __device__ inline static void run
    (double* array, double* out, double(*func)(double, double))
    {
        // only need to synch if not working within a warp
        if (offset > 16)
        {
            __syncthreads();
        }

        // only continue if it's in the lower half
        if (threadIdx.x < offset)
        {
            array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
            Reduce< offset/2 >::run(array, out, func);
        }
    }

    __device__ inline static void run
    (int* array, int* out, int(*func)(int, int))
    {
        // only need to synch if not working within a warp
        if (offset > 16)
        {
            __syncthreads();
        }

        // only continue if it's in the lower half
        if (threadIdx.x < offset)
        {
            array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
            Reduce< offset/2 >::run(array, out, func);
        }
    }
};

template < >
class Reduce < 0 >
{
public:
    __device__ inline static void run
    (double* array, double* out, double(*func)(double, double))
    {
        out[blockIdx.x] = array[0];
    }

    __device__ inline static void run
    (int* array, int* out, int(*func)(int, int))
    {
        out[blockIdx.x] = array[0];
    }
};

#endif

