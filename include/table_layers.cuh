#ifndef CUHASH_TABLE_LAYOUT
#define CUHASH_TABLE_LAYOUT
#include <cstring>
#include <helper.cuh>
#include <limits.h>

#define key_type int32_t
#define val_type int32_t
#define Empty 0
#define Reserved INT_MAX

#define warpSize 32

__device__ auto hash = XXH32_avalanche<key_type>;

GLOBALQUALIFIER void ll_batch_insert(key_type *data, val_type *result,
                                     key_type *table_key_device,
                                     val_type *table_value_device, size_t size,size_t array_size);

GLOBALQUALIFIER void ll_batch_find(key_type *data, val_type *result,
                                   key_type *table_key_device,
                                   val_type *table_value_device, size_t size,size_t array_size);

GLOBALQUALIFIER void ht_batch_insert(key_type *data, val_type *result,
                                     key_type *table_key_device,
                                     val_type *table_value_device, size_t size,size_t array_size,
                                     size_t num_searches);

GLOBALQUALIFIER void ht_batch_find(key_type *data, val_type *result,
                                   key_type *table_key_device,
                                   val_type *table_value_device, size_t size,size_t array_size,
                                   size_t num_searches);

struct BatchProdCons {
  uint32_t _back;
  uint32_t _front;
  uint32_t in_use;
  const uint32_t capacity;
  int mul;

  uint32_t size_of_query;
  uint32_t size_of_buffer;

  int blockSize;
  int minGridSize;

  cudaStream_t *stream;
  cudaEvent_t *evt;

  key_type *query_host;
  key_type *query_device;

  val_type *result_device;
  val_type *result_host;

  BatchProdCons(uint32_t size = 5, int mul = 20)
      : _back(0), _front(0), in_use(0), capacity(size),mul(mul) {

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    checkCuda(cudaOccupancyMaxPotentialBlockSize(
        &this->minGridSize, &this->blockSize, ll_batch_find, 0, 0));
    this->size_of_query = this->minGridSize * this->blockSize / warpSize * mul;
    this->size_of_buffer = this->size_of_query * capacity;

    this->stream = new cudaStream_t[capacity];
    this->evt = new cudaEvent_t[capacity];
    for (uint32_t i = 0; i < capacity; i++) {
      cudaStreamCreate(&stream[i]);
      cudaEventCreateWithFlags(&evt[i], cudaEventDisableTiming);
    }

    cudaEventRecord(evt[capacity - 1], stream[capacity - 1]);

    checkCuda(cudaMallocHost((void **)&query_host,
                             this->size_of_buffer * sizeof(key_type)));
    checkCuda(cudaMallocHost((void **)&result_host,
                             this->size_of_buffer * sizeof(val_type)));

    checkCuda(cudaMalloc((void **)&query_device,
                         this->size_of_buffer * sizeof(key_type)));
    checkCuda(cudaMalloc((void **)&result_device,
                         this->size_of_buffer * sizeof(val_type)));
  }

  ~BatchProdCons() {

    for (uint32_t i = 0; i < capacity; i++) {
      cudaStreamDestroy(stream[i]);
      cudaEventDestroy(evt[i]);
    }
    delete[] stream;
    delete[] evt;

    checkCuda(cudaFreeHost(this->query_host));
    checkCuda(cudaFreeHost(this->result_host));

    checkCuda(cudaFree(this->query_device));
    checkCuda(cudaFree(this->result_device));
  }

  void h2d(size_t loc, bool query) {
    cudaStreamSynchronize(stream[loc]);
    int offset = loc * this->size_of_query;
    if (query) {
      checkCuda(cudaMemcpyAsync(&this->query_device[offset],
                                &this->query_host[offset],
                                this->size_of_query * sizeof(key_type),
                                cudaMemcpyHostToDevice, stream[loc]));
    } else {
      checkCuda(cudaMemcpyAsync(&this->result_device[offset],
                                &this->result_host[offset],
                                this->size_of_query * sizeof(val_type),
                                cudaMemcpyHostToDevice, stream[loc]));
    }
  }

  void d2h(size_t loc, bool query) {
    int offset = loc * this->size_of_query;
    if (query) {
      checkCuda(cudaMemcpyAsync(&this->query_host[offset],
                                &this->query_device[offset],
                                this->size_of_query * sizeof(key_type),
                                cudaMemcpyDeviceToHost, stream[loc]));
    } else {
      checkCuda(cudaMemcpyAsync(&this->result_host[offset],
                                &this->result_device[offset],
                                this->size_of_query * sizeof(val_type),
                                cudaMemcpyDeviceToHost, stream[loc]));
    }
  }

  uint32_t push(key_type *keys, size_t n) {
    while (in_use == capacity) {
    }

    int offset = _front * size_of_query;
    for (size_t i = 0; i < n; i++) {
      query_host[i + offset] = keys[i];
    }
    h2d(_front, true);
    // checkCuda(cudaStreamSynchronize(stream[_front]));

    uint32_t temp = _front;

#if defined(DEBUG) || defined(_DEBUG)
    std::cout << "Adding to front " << _front << std::endl;
#endif

    _front++;
    _front %= capacity;
    ++in_use;

    return temp;
  }

  uint32_t push(key_type *keys, val_type *values, size_t n) {
    while (in_use == capacity) {
    }

    int offset = _front * size_of_query;
    for (size_t i = 0; i < n; i++) {
      query_host[i + offset] = keys[i];
      result_host[i + offset] = values[i];
    }

    h2d(_front, true);
    h2d(_front, false);
    // checkCuda(cudaStreamSynchronize(stream[_front]));

#if defined(DEBUG) || defined(_DEBUG)
    std::cout << "Adding to front " << _front << std::endl;
#endif

    ++in_use;
    uint32_t temp = _front;
    _front++;
    _front %= capacity;

    return temp;
  }

  void pop(bool query) {

    d2h(_back, query);
    checkCuda(cudaStreamSynchronize(stream[_back]));

#if defined(DEBUG) || defined(_DEBUG)
    std::cout << "Removing from " << _back << std::endl;
#endif

    _back++;
    _back %= capacity;
    --in_use;
  }

  friend std::ostream &operator<<(std::ostream &out, const BatchProdCons &cb) {
    for (unsigned i = cb._back, count = 0; count != cb.in_use;
         i = (i + 1) % cb.capacity, count++) {
      // out << cb.data[i] << " ";
    }
    return out;
  }
};

struct LLlayer {
  uint32_t size;
  key_type *table_key_device;
  val_type *table_value_device;

  HOSTQUALIFIER INLINEQUALIFIER explicit LLlayer(uint32_t size) {
    this->size = size;
    checkCuda(cudaMalloc((void **)&table_key_device, size * sizeof(key_type)));
    checkCuda(
        cudaMalloc((void **)&table_value_device, size * sizeof(val_type)));
    checkCuda(cudaMemset(table_key_device, Empty, size * sizeof(key_type)));
  }
  HOSTQUALIFIER INLINEQUALIFIER ~LLlayer() {
    checkCuda(cudaFree(this->table_key_device));
    checkCuda(cudaFree(this->table_value_device));
  }
};

struct HTLayer {

  uint32_t size;
  uint32_t num_searches;
  key_type *table_key_device;
  val_type *table_value_device;

  HOSTQUALIFIER INLINEQUALIFIER explicit HTLayer(uint32_t size) {
    this->size = size;
    checkCuda(cudaMalloc((void **)&table_key_device, size * sizeof(key_type)));
    checkCuda(
        cudaMalloc((void **)&table_value_device, size * sizeof(val_type)));
    checkCuda(cudaMemset(table_key_device, Empty, size * sizeof(key_type)));
  }
  HOSTQUALIFIER INLINEQUALIFIER ~HTLayer() {
    checkCuda(cudaFree(this->table_key_device));
    checkCuda(cudaFree(this->table_value_device));
  }
};

// class HTLayer
// {
//     uint32_t size;
//     uint32_t num_searches;
//     key_type *table_key_device;
//     val_type *table_value_device;

// public:
//     HOSTDEVICEQUALIFIER INLINEQUALIFIER void batch_insert(const key_type
//     *data, size_t n, val_type *result)
//     {
//         key_type datum = data[n / warpSize];
//         size_t key = hash(datum);
//         for (size_t i = 0; i < num_searches; i++)
//         {
//             size_t loc = threadIdx.x + key;
//             int warp_index = threadIdx.x % warpSize;
//             size_t leader = __ffs(__ballot_sync(FULL_MASK,
//             table_key_device[loc] == Empty)); if (leader != 0)
//             {
//                 if (leader == (warp_index - 1) &&
//                 atomicCAS(table_key_device[loc], Empty, datum))
//                 {
//                     data[n / warpSize] = Empty;
//                     table_value_device[loc] = result[n / warpSize];
//                 }
//                 break;
//             }
//         }
//     }

//     HOSTDEVICEQUALIFIER INLINEQUALIFIER void
//     batch_find(const key_type *data, size_t n, val_type *result)
//     {
//         key_type datum = data[n / warpSize];
//         size_t key = hash(datum);
//         size_t loc = threadIdx.x + key;
//         for (int i = 0; i < num_searches; i++)
//         {
//         //Use shared memory to store flag data?

//             size_t loc = threadIdx.x + key + i * warpSize;
//             if (table_key_device[key] == datum &&
//             atomicCAS(table_key_device[key], datum, Reserved))
//             {
//                 result[n / warpSize] = table_value_device[key];
//                 data[n / warpSize] = Empty;
//                 table_key_device[key] = datum;
//             }
//             __warpsync();
//             if(data[n / warpSize] == Empty)
//             {
//                 break;
//             }
//         }
//     }

//     HOSTQUALIFIER INLINEQUALIFIER explicit HTLayer(uint32_t size = 1000,
//     uint32_t num_searches = 2, cudaDeviceProp *prop = nullptr)
//     {
//         this->size = size;
//         this->num_searches = num_searches;
//         checkCuda(cudaMalloc((void **)&table_key_device, size *
//         sizeof(key_type))); checkCuda(cudaMalloc((void
//         **)&table_value_device, size * sizeof(val_type)));
//         checkCuda(cudaMemset(table_key_device, Empty, size *
//         sizeof(key_type)));
//     }

//     HOSTQUALIFIER INLINEQUALIFIER ~HTLayer()
//     {
//         checkCuda(cudaFree(this->table_key_device));
//         checkCuda(cudaFree(this->table_value_device));
//     }
// };

// class LargeLayer
// {
//     uint32_t num_buckets;
//     key_type *table_key_device;
//     val_type *table_value_device;

// public:
//     LargeLayer(uint32_t num_buckets = 256, cudaDeviceProp *prop = nullptr)
//     {
//         this->num_buckets = num_buckets;
//         checkCuda(cudaMalloc((void **)&table_key_device, warpSize *
//         num_buckets * sizeof(key_type))); checkCuda(cudaMalloc((void
//         **)&table_value_device, warpSize * num_buckets * sizeof(val_type)));
//         checkCuda(cudaMemset(table_key_device, Empty, warpSize * num_buckets
//         * sizeof(key_type)));
//     }

//     ~LargeLayer()
//     {
//         checkCuda(cudaFree(this->table_key_device));
//         checkCuda(cudaFree(this->table_value_device));
//     }
// };
#endif

// https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memcpy_async_pipeline
// https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
// Each block has its own shared memory
// Transfer data from host to shared and then use.
// sharedMemPerBlock
// warpSize
