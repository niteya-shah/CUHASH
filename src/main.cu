#include <iostream>
#include <experimental/random>
#include <string>
#include <CUHASH.cuh>


int main(int argc, char *argv[])
{
	CUHASH *x = new CUHASH();
	int n = x->batch->blockSize * x->batch->minGridSize/warpSize;
	key_type *key = new key_type[n];
	val_type *val = new val_type[n];
	int misses, K = 1;
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
		
		x->batch_insert(key, val, n);

		// for(int i = 0;i < n;i++)
		// {
		// 	printf("%i: %i\n", x->batch->query_host[i], val[i]);
		// }

		int *result = x->batch_find(key, n);

		for(int i = 0;i < n;i++)
		{
			// printf("%i: %i, %i\n", key[i], val[i], result[i]);
			if(val[i] != result[i])
			{
				printf("%i: %i, %i\n", key[i], val[i], result[i]);
				misses++;
			}

		}
		counter[k] = misses;
	}
	for(int k = 0;k < K;k++)
		printf("%i ",counter[k]);

	delete x;
}