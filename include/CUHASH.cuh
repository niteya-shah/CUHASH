#ifndef CUHASH_HPP
#define CUHASH_HPP

#include <helper.cuh>
#include <table_layers.cuh>

struct CUHASH {
  LLlayer *llayer;
  HTLayer *htlayer;
  // LargeLayer *large_layer;
  BatchProdCons *batch;

  CUHASH();
  ~CUHASH();
  val_type *batch_find(key_type *key, int n);
  key_type *batch_insert(key_type *key, val_type *value, int n);
};

#endif