#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include <cstddef>



namespace kmeans_cuda {

void set_seed(unsigned int const seed);

void kmeans(
    size_t const number_of_samples,
    size_t const number_of_clusters,
    size_t const number_of_blocks_per_grid,
    size_t const number_of_threads_per_block
);

}



#endif //K_MEANS_HPP
