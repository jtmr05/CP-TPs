#ifndef K_MEANS_H
#define K_MEANS_H

#include <stddef.h>



void set_seed(unsigned int const seed);

void kmeans(size_t const NUMBER_OF_SAMPLES, size_t const NUMBER_OF_CLUSTERS, size_t const NUMBER_OF_THREADS);



#endif //K_MEANS_H
