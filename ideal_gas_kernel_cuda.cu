
#include "ftocmacros.h"
#include "cuda_common.cu"

#include "chunk_cuda.cu"
extern CloverleafCudaChunk chunk;

__global__ void device_ideal_gas_kernel_cuda
(int x_min,int x_max,int y_min,int y_max,
const double * __restrict const density,
const double * __restrict const energy,
      double * __restrict const pressure,
      double * __restrict const soundspeed)
{
    __kernel_indexes;
    double v, pressurebyenergy, cell_pressure, pressurebyvolume, sound_speed_squared;

    if(row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        v = 1.0 / density[THARR2D(0, 0, 0)];

        pressure[THARR2D(0, 0, 0)] = (1.4 - 1.0) * density[THARR2D(0, 0, 0)] * energy[THARR2D(0, 0, 0)];

        pressurebyenergy = (1.4 - 1.0) * density[THARR2D(0, 0, 0)];

        pressurebyvolume = - density[THARR2D(0, 0, 0)] * pressure[THARR2D(0, 0, 0)];

        sound_speed_squared = v * v 
            * (pressure[THARR2D(0, 0, 0)] * pressurebyenergy - pressurebyvolume);

        soundspeed[THARR2D(0, 0, 0)] = sqrt(sound_speed_squared);
    }
}

extern "C" void ideal_gas_kernel_cuda_
(int *x_min,int *x_max,int *y_min,int *y_max, int* predict,
const double *density, const double *energy, double *pressure, double *soundspeed)
{
    chunk.ideal_gas_kernel(*predict);
}

void CloverleafCudaChunk::ideal_gas_kernel
(int predict)
{
    _CUDA_BEGIN_PROFILE_name(device);
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
    _CUDA_END_PROFILE_name(device);
}
