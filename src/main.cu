#include <iostream>
#include <experimental/random>
#include <string>
#include <CUHASH.cuh>

int main(int argc, char *argv[])
{
	CUHASH *x = new CUHASH();
	int key[32];
	int val[32];

	for(int i = 0;i < 32;i++)
	{
		key[i] = std::experimental::randint(0, 10000);
		val[i] = std::experimental::randint(0, 10000);
	}
	
	x->batch_insert(key, val, 32);

	int *result = x->batch_find(key, 32);
	for(int i = 0;i < 32;i++)
	{
		printf("%i: %i, %i\n", key[i], val[i], result[i]);
	}

	delete x;
}