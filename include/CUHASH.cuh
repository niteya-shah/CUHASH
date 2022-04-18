#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <cooperative_groups.h>
#include <helper.cuh>
#include <table_layers.cuh>

GLOBALQUALIFIER void  batch_insert() {
        
}


class CUHASH{
    LLlayer *llayer;
    // HTLayer *htlayer;
    // LargeLayer *large_layer;
    BatchProdCons *batch;
    public:
    CUHASH(){
        this->llayer = new LLlayer();
        // this->htlayer = new HTLayer();
        // this->large_layer = new LargeLayer();
        this->batch = new BatchProdCons(); 
    }

    void batch_insert()
    {
//Occupancy for cuda https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
        key_type key[16] = {1, 2 ,3 ,4, 5, 6, 7, 8, 9 , 10, 11, 12, 13, 14, 15, 16};
        val_type val[16] = {1, 2 ,3 ,4, 5, 6, 7, 8, 9 , 10, 11, 12, 13, 14, 15, 16};
        int n = 16;
        cudaMemcpy(this->query_device + loc * this->size_of_query, &key, this->size_of_query, cudaMemcpyHostToDevice);
        cudaMemcpy(this->result_device + loc * this->size_of_query, &val, this->size_of_query, cudaMemcpyHostToDevice);
        batch_insert<<<1,512>>>(this->query_device + loc * this->size_of_query, this->result_device + loc * this->size_of_query);
    }

    ~CUHASH(){
        delete this->llayer;
        delete this->htlayer;
        delete this->large_layer;
        delete this->batch;
    }
};

#endif