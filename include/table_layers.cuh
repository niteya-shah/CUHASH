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

__device__ auto hash = XXH32_avalanche<key_type>;

GLOBALQUALIFIER void ll_batch_insert(key_type *data, val_type *result, key_type* table_key_device, val_type* table_value_device, size_t size)
{
    size_t n =  blockIdx.x * blockDim.x + threadIdx.x;
    key_type datum = data[n / warpSize];
    size_t key = hash(datum);
    int warp_index = threadIdx.x % warpSize;
    size_t loc = (warp_index + key)%size;

    size_t leader = __ffs(__ballot_sync(FULL_MASK, table_key_device[loc] == Empty));

    if (leader == (warp_index + 1) && Empty == atomicCAS(&table_key_device[loc], Empty, datum))
    {
        data[n / warpSize] = Empty;
        table_value_device[loc] = result[n / warpSize];
    }
}

GLOBALQUALIFIER void ll_batch_find(key_type *data, val_type *result, key_type* table_key_device, val_type* table_value_device, size_t size)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    key_type datum = data[n / warpSize];
    size_t key = hash(datum);
    int warp_index = threadIdx.x % warpSize;
    size_t loc = (warp_index + key)%size;

    if (table_key_device[loc] == datum && datum == atomicCAS(&table_key_device[loc], datum, Reserved))
    {
        result[n / warpSize] = table_value_device[loc];
        data[n / warpSize] = Empty;
        table_key_device[loc] = datum;
    }
}

GLOBALQUALIFIER void ht_batch_insert(key_type *data, val_type *result, key_type* table_key_device, val_type* table_value_device, size_t size, size_t num_searches = 5)
{
    size_t n =  blockIdx.x * blockDim.x + threadIdx.x;
    key_type datum = data[n / warpSize];
    
    if(datum == Empty)
    {
        return;
    }
    
    size_t key = hash(datum);
    int warp_index = threadIdx.x % warpSize;
    int flag = 0;
    for(int i = 0;i < num_searches;i++)
    {
        size_t loc = (warp_index + key + i * warpSize)%size;

        size_t leader = __ffs(__ballot_sync(FULL_MASK, table_key_device[loc] == Empty));
        if(leader != 0)
        {
            if (leader == (warp_index + 1) && Empty == atomicCAS(&table_key_device[loc], Empty, datum))
            {
                data[n / warpSize] = Empty;
                table_value_device[loc] = result[n / warpSize];
                flag = 1;
            }
            __syncwarp();
            flag = __shfl_sync(FULL_MASK, flag, leader - 1);
            if(flag)
                return;
        }
    }
}

GLOBALQUALIFIER void ht_batch_find(key_type *data, val_type *result, key_type* table_key_device, val_type* table_value_device, size_t size, size_t num_searches = 5)
{
    __shared__ int shmem[32];
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    key_type datum = data[n / warpSize];
 
    if(datum == Empty)
    {
        return;
    }
 
    size_t key = hash(datum);
    int warp_index = threadIdx.x % warpSize;

    if (!warp_index)
    {
        shmem[threadIdx.x/warpSize] = 0;
    }

    for(int i = 0;i < num_searches;i++)
    {
        size_t loc = (warp_index + key + i * warpSize)%size;

        if (table_key_device[loc] == datum && datum == atomicCAS(&table_key_device[loc], datum, Reserved))
        {
            result[n / warpSize] = table_value_device[loc];
            data[n / warpSize] = Empty;
            table_key_device[loc] = datum;
            shmem[threadIdx.x/warpSize] = 1;
        }
        __syncwarp();
        if(shmem[threadIdx.x/warpSize])
            return;
    }
}

struct BatchProdCons
{
    uint32_t num_batches;
    uint32_t size_of_query;
    uint32_t size_of_buffer;

    int blockSize;   
    int minGridSize;

    int loc;
    cudaStream_t *stream;


    key_type *query_host;
    key_type *query_device;

    val_type *result_device;
    val_type *result_host;

    int get_loc()
    {
        this->loc = (loc + 1)%this->num_batches;
        return this->loc;
    }

    void h2d(size_t loc, bool query)
    {
        if (query)
        {
            checkCuda(cudaMemcpyAsync(this->query_device + loc * this->size_of_query, this->query_host + loc * this->size_of_query, this->size_of_query, cudaMemcpyHostToDevice, stream[loc]));
        }
        else
        {
            checkCuda(cudaMemcpyAsync(this->result_device + loc * this->size_of_query, this->result_host + loc * this->size_of_query, this->size_of_query, cudaMemcpyHostToDevice, stream[loc]));
        }
    }

    void d2h(size_t loc, bool query)
    {
        if (query)
        {
            checkCuda(cudaMemcpyAsync(this->query_host + loc * this->size_of_query, this->query_device + loc * this->size_of_query, this->size_of_query, cudaMemcpyDeviceToHost, stream[loc]));
        }
        else
        {
            checkCuda(cudaMemcpyAsync(this->result_host + loc * this->size_of_query, this->result_device + loc * this->size_of_query, this->size_of_query, cudaMemcpyDeviceToHost, stream[loc]));
        }
    }

    BatchProdCons(uint32_t num_batches = 5)
    {
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, 0));

        checkCuda(cudaOccupancyMaxPotentialBlockSize(&this->minGridSize, &this->blockSize, ll_batch_find, 0, 0));
        this->size_of_query = this->minGridSize * this->blockSize * sizeof(key_type) / warpSize;
        this->size_of_buffer = this->size_of_query * num_batches;
        this->num_batches = num_batches;

        this->loc = 0;
        this->stream = new cudaStream_t[num_batches];
        for(uint32_t i = 0;i < num_batches;i++)
        {
            cudaStreamCreate(&stream[i]);
        }

        checkCuda(cudaMallocHost((void **)&query_host, this->size_of_buffer));
        checkCuda(cudaMallocHost((void **)&result_host, this->size_of_buffer));

        checkCuda(cudaMalloc((void **)&query_device, this->size_of_buffer));
        checkCuda(cudaMalloc((void **)&result_device, this->size_of_buffer));
    }
    ~BatchProdCons()
    {
        for(uint32_t i = 0;i < num_batches;i++)
        {
            cudaStreamDestroy(stream[i]);
        }
        delete[] stream;

        checkCuda(cudaFreeHost(this->query_host));
        checkCuda(cudaFreeHost(this->result_host));

        checkCuda(cudaFree(this->query_device));
        checkCuda(cudaFree(this->result_device));
    }
};

struct LLlayer
{
    uint32_t size;
    key_type *table_key_device;
    val_type *table_value_device;

    HOSTQUALIFIER INLINEQUALIFIER explicit LLlayer(uint32_t size = 1000)
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

struct HTLayer
{

    uint32_t size;
    uint32_t num_searches;
    key_type *table_key_device;
    val_type *table_value_device;

    HOSTQUALIFIER INLINEQUALIFIER explicit HTLayer(uint32_t size = 100000)
    {
        this->size = size;
        checkCuda(cudaMalloc((void **)&table_key_device, size * sizeof(key_type)));
        checkCuda(cudaMalloc((void **)&table_value_device, size * sizeof(val_type)));
        checkCuda(cudaMemset(table_key_device, Empty, size * sizeof(key_type)));
    }
    HOSTQUALIFIER INLINEQUALIFIER ~HTLayer()
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
