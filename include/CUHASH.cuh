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

        key_type *qd = (key_type*)this->batch->allocate_data(key, 0, true);
        val_type *rd = (val_type*)this->batch->allocate_data(val, 0, false);
        batch_insert<<<1, 512>>>(qd, rd);
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