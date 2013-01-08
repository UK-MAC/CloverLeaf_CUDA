#include "cuda_common.cu"

/********************/

// could put this check in, prob doesnt need it
// if(row > 1 - depth && row < y_max + 2 + depth + y_inc)

// left/right buffer
// index=j+(k+depth-1)*depth

// left index 
// left_snd_buffer(index)=field(chunks(chunk)%field%x_min+x_inc-1+j,k)
// field(chunks(chunk)%field%x_min-j,k)=left_rcv_buffer(index)

// right index
// right_snd_buffer(index)=field(chunks(chunk)%field%x_max+1-j,k)
// field(chunks(chunk)%field%x_max+x_inc+j,k)=right_rcv_buffer(index)

/********************/

// top/bottom buffer
// index=j+depth+(k-1)*(chunks(chunk)%field%x_max+x_inc+(2*depth))

// bottom index
// bottom_snd_buffer(index)=field(j,chunks(chunk)%field%y_min+y_inc-1+k)
// field(j,chunks(chunk)%field%y_min-k)=bottom_rcv_buffer(index)

// top index
// top_snd_buffer(index)=field(j,chunks(chunk)%field%y_max+1-k)
// field(j,chunks(chunk)%field%y_max+y_inc+k)=top_rcv_buffer(index)

/********************/

// j is column
// k is row

// pack buffers

// >= is more like fortran do loops, should help with conversion
#define BETWEEN_COLUMNS(upper, lower) (column >= (lower) && column <= (upper))
#define BETWEEN_ROWS(upper, lower) (row >= (lower) && row <= (upper))

__global__ void device_packLeftBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* left_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // x_min + x_inc - 1 + j
    // x_min = 1 in fortran -> 2 in c
    // j = 1 or 2
    if(column == 2 + x_inc - 1 + 1)
    {
        left_buffer[row] = array[THARR2D(0, 0, x_extra)];
    }
    else if(depth > 1 && column == 2 + x_inc - 1 + 2)
    {
        left_buffer[row + y_max + 4 + y_extra] = array[THARR2D(0, 0, x_extra)];
    }
}

__global__ void device_unpackLeftBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* left_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // x_min - j
    if(column == 2 - 1)
    {
        array[THARR2D(0, 0, x_extra)] = left_buffer[row];
    }
    else if(depth > 1 && column == 2 - 2)
    {
        array[THARR2D(0, 0, x_extra)] = left_buffer[row + y_max + 4 + y_extra];
    }
}

/************************************************************/

__global__ void device_packRightBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* right_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // x_max + 1 - j
    if(column == x_max + 1 + 1 - 1)
    {
        right_buffer[row] = array[THARR2D(0, 0, x_extra)];
    }
    else if(depth > 1 && column == x_max + 1 + 1 - 2)
    {
        //array[THARR2D(0, 0, x_extra)] = right_buffer[row + y_max + 4];
        right_buffer[row + y_max + 4 + y_extra] = array[THARR2D(0, 0, x_extra)];
    }
}

__global__ void device_unpackRightBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* right_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // x_max + x_inc + j
    if(column == x_max + 1 + x_inc + 1)
    {
        array[THARR2D(0, 0, x_extra)] = right_buffer[row];
    }
    else if(depth > 1 && column == x_max + 1 + x_inc + 1)
    {
        array[THARR2D(0, 0, x_extra)] = right_buffer[row + y_max + 4 + y_extra];
    }
}

/************************************************************/

__global__ void device_packBottomBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* bottom_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // y_min + y_inc - 1 + k
    if(row == 2 + y_inc - 1 + 1)
    {
        bottom_buffer[row] = array[THARR2D(0, 0, x_extra)];
    }
    else if(depth > 1 && row == 2 + y_inc - 1 + 2)
    {
        bottom_buffer[row + x_max + 4 + x_extra] = array[THARR2D(0, 0, x_extra)];
    }
}

__global__ void device_unpackBottomBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* bottom_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // y_min - k
    if(row == 2 - 1)
    {
        array[THARR2D(0, 0, x_extra)] = bottom_buffer[row];
    }
    else if(depth > 1 && row == 2 - 2)
    {
        array[THARR2D(0, 0, x_extra)] = bottom_buffer[row + x_max + 4 + x_extra];
    }
}

/************************************************************/

__global__ void device_packTopBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* top_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // y_min + y_inc - 1 + k
    if(row == 2 + y_inc - 1 + 1)
    {
        top_buffer[row] = array[THARR2D(0, 0, x_extra)];
    }
    else if(depth > 1 && row == 2 + y_inc - 1 + 2)
    {
        top_buffer[row + x_max + 4 + x_extra] = array[THARR2D(0, 0, x_extra)];
    }
}

__global__ void device_unpackTopBuffer
(int x_min,int x_max,int y_min,int y_max,
int x_extra, int y_extra,
int x_inc, int y_inc,
double* array,
double* top_buffer,
int buffer_size,
int depth)
{
    __large_kernel_indexes;

    // y_max + y_inc + k
    if(row == y_max + 1 + y_inc + 1)
    {
        array[THARR2D(0, 0, x_extra)] = top_buffer[row];
    }
    else if(depth > 1 && row == y_max + 1 + y_inc + 2)
    {
        array[THARR2D(0, 0, x_extra)] = top_buffer[row + x_max + 4 + x_extra];
    }
}

