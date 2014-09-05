
__global__ void device_revert_kernel_cuda
(int x_min, int x_max, int y_min, int y_max,
const double* __restrict const density0,
      double* __restrict const density1,
const double* __restrict const energy0,
      double* __restrict const energy1)
{
    __kernel_indexes;

    if (row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)];
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)];
    }
}

