
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

extern CudaDevPtrStorage pointer_storage;

__global__ void device_ideal_gas_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double * __restrict const density,
const double * __restrict const energy,
      double * __restrict const pressure,
      double * __restrict const soundspeed)
{
    __kernel_indexes;

    if(row > 1 && row < y_max+2
    && column > 1 && column < x_max+2)
    {
        double pressurebyenergy = (1.4 - 1.0) * density[THARR2D(0, 0, 0)];
        double cell_pressure = pressurebyenergy * energy[THARR2D(0, 0, 0)];
        double pressurebyvolume = cell_pressure 
            * ( - density[THARR2D(0, 0, 0)]);
        double v = 1.0/density[THARR2D(0, 0, 0)];
        double sound_speed_squared = v * v 
            * (cell_pressure * pressurebyenergy - pressurebyvolume);
        pressure[THARR2D(0, 0, 0)] = cell_pressure;
        soundspeed[THARR2D(0, 0, 0)] = sqrt(sound_speed_squared);
    }
}

void ideal_gas_cuda
( int x_min,int x_max,int y_min,int y_max,
const double *density,
const double *energy,
double *pressure,
double *soundspeed)
{

    pointer_storage.setSize(x_max, y_max);

    double* density_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, density, BUFSZ2D(0, 0));
    double* energy_d = pointer_storage.getDevStorageAndCopy(__LINE__, __FILE__, energy, BUFSZ2D(0, 0));
    double* soundspeed_d = pointer_storage.getDevStorage(__LINE__, __FILE__);
    double* pressure_d = pointer_storage.getDevStorage(__LINE__, __FILE__);

#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(device);
#endif
    device_ideal_gas_kernel_cuda<<< ((x_max+4)*(y_max+4))/BLOCK_SZ, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density_d, energy_d, pressure_d, soundspeed_d);
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(device);
#endif

    errChk(__LINE__, __FILE__);

    pointer_storage.freeDevStorage(energy_d);
    pointer_storage.freeDevStorage(density_d);
    pointer_storage.freeDevStorageAndCopy(soundspeed_d, soundspeed, BUFSZ2D(0, 0));
    pointer_storage.freeDevStorageAndCopy(pressure_d, pressure, BUFSZ2D(0, 0));

}

extern "C" void ideal_gas_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max, int* predict,
const double *density, const double *energy, double *pressure, double *soundspeed)
{
#ifdef TIME_KERNELS
_CUDA_BEGIN_PROFILE_name(host);
#endif
    #ifndef CUDA_RESIDENT
    ideal_gas_cuda(*x_min,*x_max,*y_min,*y_max, density, energy, pressure, soundspeed);
    #else
    chunk.ideal_gas_kernel(*predict);
    #endif
#ifdef TIME_KERNELS
_CUDA_END_PROFILE_name(host);
#endif
}

// device resident definition
void CloverleafCudaChunk::ideal_gas_kernel
(int predict)
{
    #ifdef TIME_KERNELS
    _CUDA_BEGIN_PROFILE_name(device);
    #endif
    if(predict)
    {
        device_ideal_gas_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min,x_max,y_min,y_max, density1, energy1, pressure, soundspeed);
        errChk(__LINE__, __FILE__);
    }
    else
    {
        device_ideal_gas_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
        (x_min,x_max,y_min,y_max, density0, energy0, pressure, soundspeed);
        errChk(__LINE__, __FILE__);
    }
    #ifdef TIME_KERNELS
    _CUDA_END_PROFILE_name(device);
    #endif
}
