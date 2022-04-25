#include <iostream>
#include <experimental/random>
#include <string>
#include <CUHASH.cuh>

int main(int argc, char *argv[])
{
	CUHASH *x = new CUHASH();
	int n = x->batch->blockSize * x->batch->minGridSize;
	key_type *key = new key_type[n];
	val_type *val = new val_type[n];

	for(int i = 0;i < n;i++)
	{
		key[i] = std::experimental::randint(0, 10000);
		val[i] = std::experimental::randint(0, 10000);
	}
	
	x->batch_insert(key, val, n);

	// int *result = x->batch_find(key, n);
	// for(int i = 0;i < 64;i++)
	// {
	// 	printf("%i: %i, %i\n", key[i], val[i], result[i]);
	// }

	delete x;
}