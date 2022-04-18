#include <iostream>
#include <string>
#include <CUHASH.cuh>

int main(int argc, char *argv[])
{
	CUHASH<> *x = new CUHASH<>();
	delete x;
}