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

	for(int k= 0;k < 1;k++)
	{

		for(int i = 0;i < n;i++)
		{
			// key[i] = std::experimental::randint(0, INT_MAX);
			// val[i] = std::experimental::randint(0, INT_MAX);
			key[i] = i + 1;
			val[i] = i + 1;
		}
		
		x->batch_insert(key, val, n);

		for(int i = 0;i < n;i++)
		{
			key[i] = i + 1;
		}

		int *result = x->batch_find(key, n);

		for(int i = 0;i < n;i++)
		{
			if(val[i] != result[i])
				printf("%i : %i\n", key[i], result[i]);
		}
	}

	delete x;
}