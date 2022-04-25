#ifndef CUHASH_HELPERS
#define CUHASH_HELPERS

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define HOSTDEVICEQUALIFIER  __host__ __device__
#else
#define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
#define INLINEQUALIFIER  __forceinline__
#else
#define INLINEQUALIFIER inline
#endif

#ifdef __CUDACC__
#define GLOBALQUALIFIER  __global__
#else
#define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
#define DEVICEQUALIFIER  __device__
#else
#define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
#define HOSTQUALIFIER  __host__
#else
#define HOSTQUALIFIER
#endif

// Hash function finalizer from xxh3 hash
// https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h

#define XXH_PRIME32_2  0x85EBCA77U
#define XXH_PRIME32_3  0xC2B2AE3DU

#define FULL_MASK 0xffffffff

template<typename key_type>
HOSTDEVICEQUALIFIER static size_t XXH32_avalanche(key_type hash)
{
    hash ^= hash >> 15;
    hash *= XXH_PRIME32_2;
    hash ^= hash >> 13;
    hash *= XXH_PRIME32_3;
    hash ^= hash >> 16;
    return hash;
}

class GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;

public:

	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {
		cudaEventRecord(start, 0);
	}

	void Stop() {
		cudaEventRecord(stop, 0);
	}

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed / 1000;
	}

};

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

#endif
