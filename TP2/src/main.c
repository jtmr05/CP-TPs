#include "k_means.h"

#include <stdlib.h>



static size_t const NUMBER_OF_SAMPLES = 10000000;
static size_t const NUMBER_OF_CLUSTERS = 4;
static unsigned int const SEED = 10;


int main(void){

	set_seed(SEED);

	kmeans(NUMBER_OF_SAMPLES, NUMBER_OF_CLUSTERS);

	return 0;
}
