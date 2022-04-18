#ifndef CUHASH_TABLE_LAYOUT
#define CUHASH_TABLE_LAYOUT
#include <limits.h>
#include <cstring>

#include <helper.cuh>

#define key_type int32_t
#define val_type int32_t
#define Empty 0
#define Reserved INT_MAX

#define warpSize 32

auto hash = XXH32_avalanche<key_type>;

class BatchProdCons
{
    uint32_t num_batches;
    uint32_t size_of_query;
    uint32_t size_of_buffer;


    key_type *query_host;
    key_type *query_device;

    val_type *result_device;
    val_type *result_host;

public:

    void allocate_query(key_type* key, size_t loc)
    {
        cudaMemcpy(this->query_device + loc * this->size_of_query, key, this->size_of_query, cudaMemcpyHostToDevice);
    }

    BatchProdCons(uint32_t num_batches = 5)
    {
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, 0));

        uint32_t max_queries = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        if (num_batches * warpSize > max_queries)
        {
            std::cerr << "Incorrect size";
            exit(1);
        }
        this->size_of_query = warpSize * sizeof(key_type);
        this->size_of_buffer = this->size_of_query * num_batches;

        checkCuda(cudaMallocHost((void **)&query_host, size_of_buffer));
        checkCuda(cudaMallocHost((void **)&result_host, size_of_buffer));

        checkCuda(cudaMalloc((void **)&query_device, size_of_buffer));
        checkCuda(cudaMalloc((void **)&result_device, size_of_buffer));
    }
    ~BatchProdCons()
    {
        checkCuda(cudaFreeHost(this->query_host));
        checkCuda(cudaFreeHost(this->result_host));

        checkCuda(cudaFree(this->query_device));
        checkCuda(cudaFree(this->result_device));
    }
};


class LLlayer
{
    uint32_t size;
    key_type *table_key_device;
    val_type *table_value_device;

public:
    
    DEVICEQUALIFIER INLINEQUALIFIER void batch_insert(key_type *data, val_type *result)
    {
        size_t n = threadIdx.x;
        key_type datum = data[n / warpSize];
        size_t key = hash(datum);
        size_t loc = threadIdx.x + key;
        int warp_index = threadIdx.x % warpSize;
        size_t leader = __ffs(__ballot_sync(FULL_MASK, table_key_device[loc] == Empty));

        if (leader == (warp_index - 1) && Empty == atomicCAS(table_key_device[loc], Empty, datum))
        {
            data[n / warpSize] = Empty;
            table_value_device[loc] = result[n / warpSize];
        }
    }

    

    DEVICEQUALIFIER INLINEQUALIFIER void batch_find(const key_type *data, size_t n, val_type *result)
    {
        key_type datum = data[n / warpSize];
        size_t key = hash(datum);
        size_t loc = threadIdx.x + key;

        if (table_key_device[key] == datum && Reserved == atomicCAS(table_key_device[loc], datum, Reserved))
        {
            result[n / warpSize] = table_value_device[loc];
            data[n / warpSize] = Empty;
            table_key_device[loc] = datum;
        }
    }

    HOSTQUALIFIER INLINEQUALIFIER explicit LLlayer(uint32_t size = 100, cudaDeviceProp *prop = nullptr)
    {
        this->size = size;
        checkCuda(cudaMalloc((void **)&table_key_device, size * sizeof(key_type)));
        checkCuda(cudaMalloc((void **)&table_value_device, size * sizeof(val_type)));
        checkCuda(cudaMemset(table_key_device, Empty, size * sizeof(key_type)));
    }
    HOSTQUALIFIER INLINEQUALIFIER ~LLlayer()
    {
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
//     HOSTDEVICEQUALIFIER INLINEQUALIFIER void batch_insert(const key_type *data, size_t n, val_type *result)
//     {
//         key_type datum = data[n / warpSize];
//         size_t key = hash(datum);
//         for (size_t i = 0; i < num_searches; i++)
//         {
//             size_t loc = threadIdx.x + key;
//             int warp_index = threadIdx.x % warpSize;
//             size_t leader = __ffs(__ballot_sync(FULL_MASK, table_key_device[loc] == Empty));
//             if (leader != 0)
//             {
//                 if (leader == (warp_index - 1) && atomicCAS(table_key_device[loc], Empty, datum))
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
//             if (table_key_device[key] == datum && atomicCAS(table_key_device[key], datum, Reserved))
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

//     HOSTQUALIFIER INLINEQUALIFIER explicit HTLayer(uint32_t size = 1000, uint32_t num_searches = 2, cudaDeviceProp *prop = nullptr)
//     {
//         this->size = size;
//         this->num_searches = num_searches;
//         checkCuda(cudaMalloc((void **)&table_key_device, size * sizeof(key_type)));
//         checkCuda(cudaMalloc((void **)&table_value_device, size * sizeof(val_type)));
//         checkCuda(cudaMemset(table_key_device, Empty, size * sizeof(key_type)));
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
//         checkCuda(cudaMalloc((void **)&table_key_device, warpSize * num_buckets * sizeof(key_type)));
//         checkCuda(cudaMalloc((void **)&table_value_device, warpSize * num_buckets * sizeof(val_type)));
//         checkCuda(cudaMemset(table_key_device, Empty, warpSize * num_buckets * sizeof(key_type)));
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