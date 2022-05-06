#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <cooperative_groups.h>
#include <helper.cuh>
#include <table_layers.cuh>

struct CUHASH {
  LLlayer *llayer;
  HTLayer *htlayer;
  // LargeLayer *large_layer;
  BatchProdCons *batch;

  CUHASH() {
    this->llayer = new LLlayer();
    this->htlayer = new HTLayer();
    // this->large_layer = new LargeLayer();
    this->batch = new BatchProdCons();
  }

  val_type *batch_find(key_type *key, int n) {
    int loc = this->batch->push(key, n);
    int previous = (this->batch->capacity + loc - 1) % this->batch->capacity;

    cudaEventSynchronize(this->batch->evt[previous]);
    ll_batch_find<<<this->batch->minGridSize, this->batch->blockSize, 0,
                    this->batch->stream[loc]>>>(
        &this->batch->query_device[loc], &this->batch->result_device[loc],
        llayer->table_key_device, llayer->table_value_device, llayer->size);
    ht_batch_find<<<this->batch->minGridSize, this->batch->blockSize, 0,
                    this->batch->stream[loc]>>>(
        &this->batch->query_device[loc], &this->batch->result_device[loc],
        htlayer->table_key_device, htlayer->table_value_device, htlayer->size);
    cudaEventRecord(this->batch->evt[loc]);
    this->batch->pop(false);

    return &this->batch->result_host[loc];
  }

  key_type *batch_insert(key_type *key, val_type *value, int n) {
    // Occupancy for cuda
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

    int loc = this->batch->push(key, value, n);
    int offset = loc * this->batch->size_of_query;

    ll_batch_insert<<<this->batch->minGridSize, this->batch->blockSize, 0,
                      this->batch->stream[loc]>>>(
        &this->batch->query_device[offset], &this->batch->result_device[offset],
        llayer->table_key_device, llayer->table_value_device, llayer->size);
    ht_batch_insert<<<this->batch->minGridSize, this->batch->blockSize, 0,
                      this->batch->stream[loc]>>>(
        &this->batch->query_device[offset], &this->batch->result_device[offset],
        htlayer->table_key_device, htlayer->table_value_device, htlayer->size);

    this->batch->pop(true);
    checkCuda(cudaStreamSynchronize(this->batch->stream[loc]));
    return &this->batch->result_host[offset];
  }

  ~CUHASH() {
    delete this->llayer;
    delete this->htlayer;
    // delete this->large_layer;
    delete this->batch;
  }
};

#endif