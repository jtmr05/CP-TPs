#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include <cstddef>



namespace kmeans_cuda {

void set_seed(unsigned int const seed);

void kmeans(size_t const NUMBER_OF_SAMPLES, size_t const NUMBER_OF_CLUSTERS);

}



#endif //K_MEANS_HPP
