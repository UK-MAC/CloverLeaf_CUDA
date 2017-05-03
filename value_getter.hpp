#ifndef __VALUE_GETTER_INC
#define __VALUE_GETTER_INC

#include "cuda_common.hpp"

template<typename T>
class Value_Getter {
public:
	inline static T from_device(T* buffer_d, int index)
	{
	  T val;
	  cudaMemcpy(&val, &buffer_d[index], sizeof(T), cudaMemcpyDeviceToHost);
	  CUDA_ERR_CHECK;
	  return val;
	}
};

#endif //__VALUE_GETTER_INC
