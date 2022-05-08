#include <iostream>
#include <experimental/random>
#include <string>
#include <CUHASH.cuh>
#include <unistd.h>



int main(int argc, char *argv[])
{
	CUHASH *x = new CUHASH();
	int n = x->batch->blockSize * x->batch->minGridSize/warpSize;
	key_type *key = new key_type[n];
	val_type *val = new val_type[n];
	int misses, K = 100;
	int counter[K];

	for(int k= 0;k < K;k++)
	{
		misses = 0;
		for(int i = 0;i < n;i++)
		{
			key[i] = std::experimental::randint(1, INT_MAX - 1);
			val[i] = std::experimental::randint(1, INT_MAX - 1);
			// key[i] = i + 1;
			// val[i] = i + 1;
		}
		
		int *keys = x->batch_insert(key, val, n);
		int *result = x->batch_find(key, n);

		for(int i = 0;i < n;i++)
		{
			if(val[i] != result[i])
			{
				// printf("%i: %i, %i, %i\n", key[i], val[i], result[i - 1], result[i - 3]);
				misses++;
			}

		}
		counter[k] = misses;
	}
	for(int k = 0;k < K;k++)
		printf("%i ",counter[k]);

	delete x;
}