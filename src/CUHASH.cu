#ifndef CUHASH_CPP
#define CUHASH_CPP

#include <CUHASH.cuh>

GLOBALQUALIFIER void ll_batch_insert(key_type *data, val_type *result,
                                     key_type *table_key_device,
                                     val_type *table_value_device,
                                     size_t size) {
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  key_type datum = data[n / warpSize];
  int num_searches = blockDim.x / warpSize;

  if (datum == Empty) {
    return;
  }

  size_t key = hash(datum);
  int warp_index = threadIdx.x % warpSize;
  int flag = 0;
  for (int i = 0; i < num_searches; i++) {
    size_t loc = (warp_index + key + i * warpSize) % size;

    size_t leader =
        __ffs(__ballot_sync(FULL_MASK, table_key_device[loc] == Empty));
    if (leader != 0) {
      if (leader == (warp_index + 1) &&
          Empty == atomicCAS(&table_key_device[loc], Empty, datum)) {
        data[n / warpSize] = Empty;
        table_value_device[loc] = result[n / warpSize];
        flag = 1;
      }
      __syncwarp();
      flag = __shfl_sync(FULL_MASK, flag, leader - 1);
      if (flag)
        return;
    }
  }
}

GLOBALQUALIFIER void ll_batch_find(key_type *data, val_type *result,
                                   key_type *table_key_device,
                                   val_type *table_value_device, size_t size) {
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_index = threadIdx.x % warpSize;

  for (int i = 0; i < blockDim.x / warpSize; i++) {
    size_t ele = ((n / blockDim.x) * warpSize) + i;
    key_type datum = data[ele];
    size_t key = hash(datum);

    size_t loc = (warp_index + key) % size;

    if (table_key_device[loc] == datum &&
        datum == atomicCAS(&table_key_device[loc], datum, Reserved)) {
      result[ele] = table_value_device[loc];
      data[ele] = Empty;
      table_key_device[loc] = datum;
    }
  }
}

GLOBALQUALIFIER void ht_batch_insert(key_type *data, val_type *result,
                                     key_type *table_key_device,
                                     val_type *table_value_device, size_t size,
                                     size_t num_searches = 5) {
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  key_type datum = data[n / warpSize];

  if (datum == Empty) {
    return;
  }

  size_t key = hash(datum);
  int warp_index = threadIdx.x % warpSize;
  int flag = 0;
  for (int i = 0; i < num_searches; i++) {
    size_t loc = (warp_index + key + i * warpSize) % size;

    size_t leader =
        __ffs(__ballot_sync(FULL_MASK, table_key_device[loc] == Empty));
    if (leader != 0) {
      if (leader == (warp_index + 1) &&
          Empty == atomicCAS(&table_key_device[loc], Empty, datum)) {
        data[n / warpSize] = Empty;
        table_value_device[loc] = result[n / warpSize];
        flag = 1;
      }
      __syncwarp();
      flag = __shfl_sync(FULL_MASK, flag, leader - 1);
      if (flag)
        return;
    }
  }
}

GLOBALQUALIFIER void ht_batch_find(key_type *data, val_type *result,
                                   key_type *table_key_device,
                                   val_type *table_value_device, size_t size,
                                   size_t num_searches = 5) {
  __shared__ int shmem[32];
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  key_type datum = data[n / warpSize];

  if (datum == Empty) {
    return;
  }

  size_t key = hash(datum);
  int warp_index = threadIdx.x % warpSize;

  if (!warp_index) {
    shmem[threadIdx.x / warpSize] = 0;
  }

  for (int i = 0; i < num_searches; i++) {
    size_t loc = (warp_index + key + i * warpSize) % size;

    if (table_key_device[loc] == datum &&
        datum == atomicCAS(&table_key_device[loc], datum, Reserved)) {
      result[n / warpSize] = table_value_device[loc];
      data[n / warpSize] = Empty;
      table_key_device[loc] = datum;
      shmem[threadIdx.x / warpSize] = 1;
    }
    __syncwarp();
    if (shmem[threadIdx.x / warpSize])
      return;
  }
}

CUHASH::CUHASH(uint32_t ll_size = 1000, uint32_t ht_size = 100000) {
  this->llayer = new LLlayer(ll_size);
  this->htlayer = new HTLayer(ht_size);
  // this->large_layer = new LargeLayer();
  this->batch = new BatchProdCons();
}

val_type *CUHASH::batch_find(key_type *key, int n) {
  int loc = this->batch->push(key, n);
  int offset = loc * this->batch->size_of_query;
  int previous = (this->batch->capacity + loc - 1) % this->batch->capacity;

  cudaStreamWaitEvent(this->batch->stream[previous], this->batch->evt[previous],
                      0);

  ll_batch_find<<<this->batch->minGridSize, this->batch->blockSize, 0,
                  this->batch->stream[loc]>>>(
      &this->batch->query_device[offset], &this->batch->result_device[offset],
      llayer->table_key_device, llayer->table_value_device, llayer->size);
  ht_batch_find<<<this->batch->minGridSize, this->batch->blockSize, 0,
                  this->batch->stream[loc]>>>(
      &this->batch->query_device[offset], &this->batch->result_device[offset],
      htlayer->table_key_device, htlayer->table_value_device, htlayer->size);
  cudaEventRecord(this->batch->evt[loc]);
  this->batch->pop(false);

  return &this->batch->result_host[offset];
}

key_type *CUHASH::batch_insert(key_type *key, val_type *value, int n) {
  // Occupancy for cuda
  // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

  int loc = this->batch->push(key, value, n);
  int offset = loc * this->batch->size_of_query;
  int previous = (this->batch->capacity + loc - 1) % this->batch->capacity;

  cudaStreamWaitEvent(this->batch->stream[previous], this->batch->evt[previous],
                      0);

  ll_batch_insert<<<this->batch->minGridSize, this->batch->blockSize, 0,
                    this->batch->stream[loc]>>>(
      &this->batch->query_device[offset], &this->batch->result_device[offset],
      llayer->table_key_device, llayer->table_value_device, llayer->size);
  ht_batch_insert<<<this->batch->minGridSize, this->batch->blockSize, 0,
                    this->batch->stream[loc]>>>(
      &this->batch->query_device[offset], &this->batch->result_device[offset],
      htlayer->table_key_device, htlayer->table_value_device, htlayer->size);
  cudaEventRecord(this->batch->evt[loc]);

  this->batch->pop(true);

  return &this->batch->query_host[offset];
}

CUHASH::~CUHASH() {
  delete this->llayer;
  delete this->htlayer;
  // delete this->large_layer;
  delete this->batch;
}

#endif
