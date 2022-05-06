#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <cooperative_groups.h>
#include <helper.cuh>
#include <table_layers.cuh>

struct CUHASH
{
    LLlayer *llayer;
    HTLayer *htlayer;
    // LargeLayer *large_layer;
    BatchProdCons *batch;

    CUHASH()
    {
        this->llayer = new LLlayer();
        this->htlayer = new HTLayer();
        // this->large_layer = new LargeLayer();
        this->batch = new BatchProdCons();
    }

    val_type* batch_find(key_type *key, int n)
    {
        int loc = this->batch->get_loc();
        int offset = loc * this->batch->size_of_query;

        for(int i = 0;i < n;i++)
        {
            this->batch->query_host[i + offset] = key[i];
        }
        this->batch->h2d(loc, true);
        ll_batch_find<<<this->batch->minGridSize, this->batch->blockSize, 0, this->batch->stream[loc]>>>(&this->batch->query_device[offset], &this->batch->result_device[offset], llayer->table_key_device, llayer->table_value_device, llayer->size);
        ht_batch_find<<<this->batch->minGridSize, this->batch->blockSize, 0, this->batch->stream[loc]>>>(&this->batch->query_device[offset], &this->batch->result_device[offset], htlayer->table_key_device, htlayer->table_value_device, htlayer->size);
        this->batch->d2h(loc, true);
        this->batch->d2h(loc, false);
        checkCuda(cudaStreamSynchronize(this->batch->stream[loc]));
        return &this->batch->result_host[offset];
    }

    void batch_insert(key_type* key, val_type* value, int n)
    {
        // Occupancy for cuda https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

        int loc = this->batch->get_loc();        
        for(int i = 0;i<n;i++)
        {
            this->batch->query_host[i + loc * this->batch->size_of_query] = key[i];
            this->batch->result_host[i + loc * this->batch->size_of_query] = value[i];
        }

        this->batch->h2d(loc, true);
        this->batch->h2d(loc, false);
        int offset = loc * this->batch->size_of_query;
        ll_batch_insert<<<this->batch->minGridSize, this->batch->blockSize, 0, this->batch->stream[loc]>>>(&this->batch->query_device[offset], &this->batch->result_device[offset], llayer->table_key_device, llayer->table_value_device, llayer->size);
        ht_batch_insert<<<this->batch->minGridSize, this->batch->blockSize, 0, this->batch->stream[loc]>>>(&this->batch->query_device[offset], &this->batch->result_device[offset], htlayer->table_key_device, htlayer->table_value_device, htlayer->size);
        this->batch->d2h(loc, true);
        checkCuda(cudaStreamSynchronize(this->batch->stream[loc]));
    }

    ~CUHASH()
    {
        delete this->llayer;
        delete this->htlayer;
        // delete this->large_layer;
        delete this->batch;
    }
};

#endif