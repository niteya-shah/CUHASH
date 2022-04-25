#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <cooperative_groups.h>
#include <helper.cuh>
#include <table_layers.cuh>

struct CUHASH
{
    LLlayer *llayer;
    // HTLayer *htlayer;
    // LargeLayer *large_layer;
    BatchProdCons *batch;

    CUHASH()
    {
        this->llayer = new LLlayer();
        // this->htlayer = new HTLayer();
        // this->large_layer = new LargeLayer();
        this->batch = new BatchProdCons();
    }

    val_type* batch_find(key_type *key, int n)
    {
        int loc = this->batch->get_loc();

        for(int i = 0;i < n;i++)
        {
            this->batch->query_host[i + loc * this->batch->size_of_query] = key[i];
        }

        this->batch->h2d(loc, true);
        ll_batch_find<<<this->batch->minGridSize, this->batch->blockSize>>>(this->batch->query_device, this->batch->result_device, llayer->table_key_device, llayer->table_value_device, llayer->size);
        this->batch->d2h(loc, true);
        this->batch->d2h(loc, false);

        return this->batch->result_host + loc * this->batch->size_of_query;
    }

    void batch_insert(key_type* key, val_type* value, int n)
    {
        // Occupancy for cuda https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

        int loc = this->batch->get_loc();        
        for(int i = 0;i<n;i++)
        {
            std::cout<<i<<std::endl;
            this->batch->query_host[i + loc * this->batch->size_of_query] = key[i];
            // this->batch->result_host[i + loc * this->batch->size_of_query] = value[i];
        }

        // this->batch->h2d(loc, true);
        // this->batch->h2d(loc, false);
        // ll_batch_insert<<<1, n * warpSize>>>(this->batch->query_device,this->batch->result_device , llayer->table_key_device, llayer->table_value_device, llayer->size);
    }

    ~CUHASH()
    {
        delete this->llayer;
        // delete this->htlayer;
        // delete this->large_layer;
        delete this->batch;
    }
};

#endif