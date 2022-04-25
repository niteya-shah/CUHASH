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
	std::cout<<n<<std::endl;
	for(int i = 0;i < n;i++)
	{
		key[i] = std::experimental::randint(0, INT_MAX);
		val[i] = std::experimental::randint(0, INT_MAX);
	}
	
	x->batch_insert(key, val, n);
	
	int *result = x->batch_find(key, n);
	delete x;
}