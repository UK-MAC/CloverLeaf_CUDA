#include "cuda_common.cu"

//> \def common arguments into the kernels for both x and y direction
#define _SHARED_KERNEL_ARGS_                    \
    const int x_min,                            \
    const int x_max,                            \
    const int y_min,                            \
    const int y_max,                            \
    const int swp_nmbr,                         \
    const double* __restrict const volume,      \
    const double* __restrict const vol_flux_x,  \
    const double* __restrict const vol_flux_y,  \
    const double* __restrict const pre_vol,     \
          double* __restrict const density1,    \
          double* __restrict const energy1,     \
          double* __restrict const ener_flux

__global__ void device_pre_vol_kernel_x
(const int x_min, const int x_max,
const int y_min, const int y_max,
const int swp_nmbr,
      double* __restrict const pre_vol,
      double* __restrict const post_vol,
const double* __restrict const volume,
const double* __restrict const vol_flux_x,
const double* __restrict const vol_flux_y)
{
    __kernel_indexes;

    if(row < y_max + 4
    && column < x_max + 4)
    {
        if(swp_nmbr == 1)
        {
            pre_vol[THARR2D(0, 0, 0)] = volume[THARR2D(0, 0, 0)]
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)]
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];
            post_vol[THARR2D(0, 0, 1)] = pre_vol[THARR2D(0, 0, 0)]
                - (vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)]);
        }
        else
        {
            pre_vol[THARR2D(0, 0, 0)] = volume[THARR2D(0, 0, 0)]
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];
        }
    }
}

__global__ void device_ener_flux_kernel_x
(_SHARED_KERNEL_ARGS_,
const double* __restrict const vertexdx,
      double* __restrict const mass_flux_x)
{
    __kernel_indexes;

    double sigmat, sigmam, sigma3, sigma4, diffuw, diffdw, limiter;
    int upwind, donor, downwind, dif;
    const double one_by_six = 1.0/6.0;

    //
    //  if cell is within x area:
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //  ++xxxxxxxxxxxxxxxxxxx
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //
    if(row > 1 && row < y_max+2
    && column > 1 && column < x_max+4)
    {
        // if flowing right
        if(vol_flux_x[THARR2D(0, 0, 1)] > 0.0)
        {
            upwind = THARR2D(-2, 0, 0);
            donor = THARR2D(-1, 0, 0);
            downwind = THARR2D(0, 0, 0);
            dif = donor;
        }
        else
        {
            //
            //  tries to get from below, unless it would be reading from a cell
            //  which would be off the bottom, in which case read from cur cell
            //
            upwind = (column == x_max+3) ? THARR2D(0, 0, 0) : THARR2D(1, 0, 0);
            donor = THARR2D(0, 0, 0);
            downwind = THARR2D(-1, 0, 0);
            dif = upwind;
        }

        sigmat = fabs(vol_flux_x[THARR2D(0, 0, 1)]) / pre_vol[donor];
        sigma3 = (1.0 + sigmat) * (vertexdx[column] / vertexdx[dif % (x_max+5)]);
        sigma4 = 2.0 - sigmat;

        diffuw = density1[donor] - density1[upwind];
        diffdw = density1[downwind] - density1[donor];

        if(diffuw * diffdw > 0.0)
        {
            limiter = (1.0 - sigmat) * SIGN(1.0, diffdw)
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        }
        else
        {
            limiter = 0.0;
        }

        mass_flux_x[THARR2D(0, 0, 1)] = vol_flux_x[THARR2D(0, 0, 1)]
            * (density1[donor] + limiter);

        sigmam = fabs(mass_flux_x[THARR2D(0, 0, 1)])
            / (density1[donor] * pre_vol[donor]);
        diffuw = energy1[donor] - energy1[upwind];
        diffdw = energy1[downwind] - energy1[donor];

        if(diffuw * diffdw > 0.0)
        {
            limiter = (1.0 - sigmam) * SIGN(1.0, diffdw)
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        }
        else
        {
            limiter = 0.0;
        }

        ener_flux[THARR2D(0, 0, 0)] = mass_flux_x[THARR2D(0, 0, 1)]
            * (energy1[donor] + limiter);
    }
}

__global__ void device_advec_cell_kernel_x
(_SHARED_KERNEL_ARGS_,
const double* __restrict const mass_flux_x)
{
    __kernel_indexes;

    double pre_mass, post_mass, advec_vol, post_ener;

    //
    //  if cell is within x area:
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //  ++xxxxxxxxxxxxxxxxx++
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //
    if(row > 1 && row < y_max+2
    && column > 1 && column < x_max+2)
    {
        pre_mass = density1[THARR2D(0, 0, 0)] * pre_vol[THARR2D(0, 0, 0)];

        post_mass = pre_mass + mass_flux_x[THARR2D(0, 0, 1)]
            - mass_flux_x[THARR2D(1, 0, 1)];

        post_ener = (energy1[THARR2D(0, 0, 0)] * pre_mass
            + ener_flux[THARR2D(0, 0, 0)] - ener_flux[THARR2D(1, 0, 0)])
            / post_mass;

        advec_vol = pre_vol[THARR2D(0, 0, 0)] + vol_flux_x[THARR2D(0, 0, 1)]
            - vol_flux_x[THARR2D(1, 0, 1)];

        density1[THARR2D(0, 0, 0)] = post_mass / advec_vol;
        energy1[THARR2D(0, 0, 0)] = post_ener;
    }
}

//////////////////////////////////////////////////////////////////////////
//y kernels

__global__ void device_pre_vol_kernel_y
(const int x_min, const int x_max,
const int y_min, const int y_max,
const int swp_nmbr,
      double* __restrict const pre_vol,
      double* __restrict const post_vol,
const double* __restrict const volume,
const double* __restrict const vol_flux_x,
const double* __restrict const vol_flux_y)
{
    __kernel_indexes;

    if(row < y_max + 4
    && column < x_max + 4)
    {
        if(swp_nmbr == 1)
        {
            pre_vol[THARR2D(0, 0, 0)] = volume[THARR2D(0, 0, 0)]
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)]
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];
            post_vol[THARR2D(0, 0, 1)] = pre_vol[THARR2D(0, 0, 0)]
                - (vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 1)]);
        }
        else
        {
            pre_vol[THARR2D(0, 0, 0)] = volume[THARR2D(0, 0, 0)]
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];
        }
    }
}

__global__ void device_ener_flux_kernel_y
(_SHARED_KERNEL_ARGS_,
const double* __restrict const vertexdy,
      double* __restrict const mass_flux_y)
{
    __kernel_indexes;

    double sigmat, sigmam, sigma3, sigma4, diffuw, diffdw, limiter;
    int upwind, donor, downwind, dif;
    const double one_by_six = 1.0/6.0;

    //
    //  if cell is within x area:
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //  ++xxxxxxxxxxxxxxxxx++
    //  ++xxxxxxxxxxxxxxxxx++
    //
    if(row > 1 && row < y_max+4
    && column > 1 && column < x_max+2)
    {
        // if flowing right
        if(vol_flux_y[THARR2D(0, 0, 0)] > 0.0)
        {
            upwind = THARR2D(0, -2, 0);
            donor = THARR2D(0, -1, 0);
            downwind = THARR2D(0, 0, 0);
            dif = donor;
        }
        else
        {
            //
            //  tries to get from below, unless it would be reading from a cell
            //  which would be off the bottom, in which case read from cur cell
            //
            upwind = (row == y_max+3) ? THARR2D(0, 0, 0) : THARR2D(0, 1, 0);
            donor = THARR2D(0, 0, 0);
            downwind = THARR2D(0, -1, 0);
            dif = upwind;
        }

        sigmat = fabs(vol_flux_y[THARR2D(0, 0, 0)]) / pre_vol[donor];
        sigma3 = (1.0 + sigmat) * (vertexdy[row] / vertexdy[dif / (x_max+4)]);
        sigma4 = 2.0 - sigmat;

        diffuw = density1[donor] - density1[upwind];
        diffdw = density1[downwind] - density1[donor];

        if(diffuw * diffdw > 0.0)
        {
            limiter = (1.0 - sigmat) * SIGN(1.0, diffdw)
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        }
        else
        {
            limiter = 0.0;
        }

        mass_flux_y[THARR2D(0, 0, 0)] = vol_flux_y[THARR2D(0, 0, 0)]
            * (density1[donor] + limiter);

        sigmam = fabs(mass_flux_y[THARR2D(0, 0, 0)])
            / (density1[donor] * pre_vol[donor]);
        diffuw = energy1[donor] - energy1[upwind];
        diffdw = energy1[downwind] - energy1[donor];

        if(diffuw * diffdw > 0.0)
        {
            limiter = (1.0 - sigmam) * SIGN(1.0, diffdw)
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));
        }
        else
        {
            limiter = 0.0;
        }

        ener_flux[THARR2D(0, 0, 0)] = mass_flux_y[THARR2D(0, 0, 0)]
            * (energy1[donor] + limiter);
    }
}

__global__ void device_advec_cell_kernel_y
(_SHARED_KERNEL_ARGS_,
const double* __restrict const mass_flux_y)
{
    __kernel_indexes;

    double pre_mass, post_mass, advec_vol, post_ener;

    //
    //  if cell is within x area:
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //  ++xxxxxxxxxxxxxxxxx++
    //  +++++++++++++++++++++
    //  +++++++++++++++++++++
    //
    if(row > 1 && row < y_max+2
    && column > 1 && column < x_max+2)
    {
        pre_mass = density1[THARR2D(0, 0, 0)] * pre_vol[THARR2D(0, 0, 0)];

        post_mass = pre_mass + mass_flux_y[THARR2D(0, 0, 0)]
            - mass_flux_y[THARR2D(0, 1, 0)];

        post_ener = (energy1[THARR2D(0, 0, 0)] * pre_mass
            + ener_flux[THARR2D(0, 0, 0)] - ener_flux[THARR2D(0, 1, 0)])
            / post_mass;

        advec_vol = pre_vol[THARR2D(0, 0, 0)] + vol_flux_y[THARR2D(0, 0, 0)]
            - vol_flux_y[THARR2D(0, 1, 0)];

        density1[THARR2D(0, 0, 0)] = post_mass / advec_vol;
        energy1[THARR2D(0, 0, 0)] = post_ener;
    }
}
