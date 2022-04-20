#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <cooperative_groups.h>
#include <helper.cuh>
#include <table_layers.cuh>

class CUHASH
{
    LLlayer *llayer;
    // HTLayer *htlayer;
    // LargeLayer *large_layer;
    BatchProdCons *batch;

public:
    CUHASH()
    {
        this->llayer = new LLlayer();
        // this->htlayer = new HTLayer();
        // this->large_layer = new LargeLayer();
        this->batch = new BatchProdCons();
    }

    void batch_insert()
    {
        // Occupancy for cuda https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
        key_type key[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        val_type val[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

        int n = 16;
        for(int i = 0;i<n;i++)
        {
            this->batch->query_host[i] = key[i];
            this->batch->result_host[i] = val[i];
        }


        int loc = 0;
        this->batch->h2d(loc, true);
        this->batch->h2d(loc, false);

        for(int i = 0;i < n;i++)
        {
            this->batch->result_host[i] = 0;
        }

        ll_batch_insert<<<1, 512>>>(this->batch->query_device,this->batch->result_device , llayer->table_key_device, llayer->table_value_device, llayer->size);
        ll_batch_find<<<1, 512>>>(this->batch->query_device, this->batch->result_device, llayer->table_key_device, llayer->table_value_device, llayer->size);
        this->batch->d2h(loc, true);
        this->batch->d2h(loc, false);

        for(int i = 0;i < n;i++)
        {
            printf("%i : %i\n", this->batch->query_host[i], this->batch->result_host[i]);
        }
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