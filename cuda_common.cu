
#ifndef __CUDA_COMMON_INC
#define __CUDA_COMMON_INC

// size of workgroup/block
#define BLOCK_SZ 512

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

// access a volue in a 2d array given the x and y offset from current thread index, adding or subtracting a bit more if it is one of the arrays with bigger rows
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

// some arrays are larger in the x direction 
#define __large_kernel_indexes \
    const int glob_id = threadIdx.x         \
        + blockIdx.x * blockDim.x;          \
    const int row = glob_id / (x_max + 4 + x_extra);  \
    const int column = glob_id % (x_max + 4 + x_extra);

// beginning of profiling bit
#define _CUDA_BEGIN_PROFILE_name(x) \
    double x##t_0, x##t_1;          \
    x##t_0 = omp_get_wtime();

// end of profiling bit
#define _CUDA_END_PROFILE_name(x)                   \
    cudaDeviceSynchronize();                        \
    x##t_1 = omp_get_wtime();                       \
    std::cout << "[PROFILING] " << x##t_1 - x##t_0  \
    << " to calculate block \"" << #x <<            \
    "\" in " << __FILE__  <<std::endl;

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

/*******************/

#include <iostream>
#include <vector>
#include <algorithm>
#include "omp.h"

// for reduction in calc_dt and PdV
#include "thrust/copy.h"
#include "thrust/reduce.h"
#include "thrust/fill.h"
#include "thrust/functional.h"
#include "thrust/device_allocator.h"
#include "thrust/device_malloc.h"

/*******************/

// lots of time is spent error checking - define this to stop checking for errors. I don't have to tell you that this can be a silly idea
#ifndef NO_ERR_CHK

#ifdef _GNUC_
void errChk(int, std::string const&) __attribute__((always_inline));
#endif

inline void errChk
(int line_num, std::string const& file)
{
    cudaDeviceSynchronize();
    int l_e = cudaGetLastError();
    if(l_e != cudaSuccess)
    {
        std::cout << "error on line " << line_num << " of ";
        std::cout << file << std::endl;
        std::cout << "return code " << l_e; 
        switch(l_e)
        {
            case 8: std::cout << " (invalid device function - recompile correctly for architecture)"; break;
            case 11: std::cout << " (invalid value - some number was passed wrong)"; break;
            default: std::cout << " ()"; break;
        }
        std::cout << std::endl;
        exit(-1);
    }
}

#else

#define errChk(l, f) ;//nop

#endif //NO_ERR_CHK

/*******************/

class CudaDevPtrStorage
{
private:
    //work arrays used for storing data used by any kernel
    std::vector< double* > work_arrays;

    int x_max, y_max;
public:

    CudaDevPtrStorage
    ()
    :x_max(0), y_max(0)
    {
    }

    //*
    void setSize
    (int new_x_max, int new_y_max)
    {
        x_max = new_x_max;
        y_max = new_y_max;
    }
    // */

    ~CudaDevPtrStorage
    (void)
    {
        //std::for_each(work_arrays.begin(), work_arrays.end(), cudaFree);
    }

    // gets a pointer to some device storage, if it's already allocated then use it, or allocate some more
    double* getDevStorage
    (int line, std::string const& file)
    {
        double * new_storage;
        if(work_arrays.size() < 1)
        {
            cudaMalloc((void**) &new_storage, ((x_max+5)*(y_max+5)*sizeof(double)));
            errChk(line, file);
        }
        else
        {
            new_storage = work_arrays.back();
            work_arrays.pop_back();
        }
        return new_storage;
    }

    // same as above, but also copies data from an existing host pointer to the device
    double* getDevStorageAndCopy
    (int line, std::string const& file, const double* existing, size_t size)
    {
        double * new_storage;
        new_storage = getDevStorage(line, file);
        cudaMemcpy(new_storage, existing, size, cudaMemcpyHostToDevice);
        errChk(line, file);
        return new_storage;
    }

    // frees up some device storage to be used again elsewhere
    void freeDevStorage
    (double* dev_ptr)
    {
        work_arrays.push_back(dev_ptr);
    }

    // same as above, but copies data back from the device as well
    void freeDevStorageAndCopy
    (double* dev_ptr, double* existing, size_t size)
    {
        cudaMemcpy(existing, dev_ptr, size, cudaMemcpyDeviceToHost);
        errChk(__LINE__, __FILE__);
        freeDevStorage(dev_ptr);
    }
};

#endif
