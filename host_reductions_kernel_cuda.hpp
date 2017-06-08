#ifndef __HOST_REDUCTIONS_KERNEL_CUDA_INC
#define __HOST_REDUCTIONS_KERNEL_CUDA_INC

#include "cuda_common.hpp"
#include "kernel_files/reductions_kernel.cuknl"

template<typename T>
class ReduceToHost
{
private:
    inline static void reduce
    (const REDUCTION_TYPE reduction_type, T* buffer, T* result, int len)
    {
        while(len > 1)
        {
            int num_blocks = ceil(len / (double)BLOCK_SZ);
            switch(reduction_type)
            {
                case RED_SUM:
                    reduction<T, RED_SUM><<<num_blocks, BLOCK_SZ>>>(len, buffer);
                break;
                case RED_MAX:
                    reduction<T, RED_MAX><<<num_blocks, BLOCK_SZ>>>(len, buffer);
                break;
                case RED_MIN:
                    reduction<T, RED_MIN><<<num_blocks, BLOCK_SZ>>>(len, buffer);
                break;
            }
            len = num_blocks;
        }
        CUDA_ERR_CHECK;
        cudaMemcpy(result, buffer, sizeof(T), cudaMemcpyDeviceToHost);
    }

public:
    inline static void sum (T* buffer, T* result, int len)
    {
        reduce(RED_SUM, buffer, result, len);
    }

    inline static void max_element (T* buffer, T* result, int len)
    {
        reduce(RED_MAX, buffer, result, len);
    }

    inline static void min_element (T* buffer, T* result, int len)
    {
        reduce(RED_MIN, buffer, result, len);
    }
};

#endif //__HOST_REDUCTIONS_KERNEL_CUDA_INC
