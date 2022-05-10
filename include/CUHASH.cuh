#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <helper.cuh>
#include <table_layers.cuh>

struct CUHASH {
  LLlayer *llayer;
  HTLayer *htlayer;
  // LargeLayer *large_layer;
  BatchProdCons *batch;

  CUHASH(uint32_t ll_size = 1000, uint32_t ht_size = 100000);
  ~CUHASH();
  val_type *batch_find(key_type *key, int n);
  key_type *batch_insert(key_type *key, val_type *value, int n);
};

#endif