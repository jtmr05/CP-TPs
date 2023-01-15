#include "k_means.hpp"

#include <cstdio>
#include <limits>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace kmeans_cuda {

static size_t constexpr NUMBER_OF_ITERATIONS        = 20;
static float  constexpr MAX_FLOAT_VALUE             = std::numeric_limits<float>::max();


// Samples

struct Sample {

    float x, y;

    Sample() : x{0.f}, y {0.f} {}

    __device__
    Sample(float const x, float const y) : x{x}, y{y} {}
};

__device__ static inline
Sample const& operator+=(Sample& lhs, Sample const& rhs){
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

__device__ static inline
Sample const& operator/=(Sample& lhs, size_t const rhs){
    lhs.x /= rhs;
    lhs.y /= rhs;
    return lhs;
}


// Generic Vector for both Samples and Clusters

template<typename T>
struct DeviceVector {

    T* data;
    size_t const size;
    bool const is_owner;

    DeviceVector(size_t const size) : data{nullptr}, size{size}, is_owner{true} {
        cudaMalloc(&(this->data), sizeof *(this->data) * size);
    }

    DeviceVector(DeviceVector const& other) :
        data{other.data},
        size{other.size},
        is_owner{false} {}

    DeviceVector(DeviceVector&& other) = delete;

    ~DeviceVector(){
        if(this->is_owner)
            cudaFree(this->data);
    }

    void reset() const {
        cudaMemset(this->data, 0, sizeof *(this->data) * this->size);
    }
};


static inline
void fill_samples_vector(DeviceVector<Sample> const& sv){

    std::unique_ptr<Sample[]> const samples = std::make_unique<Sample[]>(sv.size);

    for (size_t i = 0; i < sv.size; ++i){

        float const x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float const y = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

        samples[i] = { x, y };
    }

    cudaMemcpy(
        sv.data,
        samples.get(),
        sizeof *samples.get() * sv.size,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    );
}

static inline
void init_clusters_vector(DeviceVector<Sample> const &centroids, DeviceVector<Sample> const &sv){
    cudaMemcpy(
        centroids.data,
        sv.data,
        sizeof *centroids.data * centroids.size,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice
    );
}


__device__ static inline
float distance_sample(Sample const s1, Sample const s2){

    float const x_diff = s1.x - s2.x;
    float const y_diff = s1.y - s2.y;

    return x_diff * x_diff + y_diff * y_diff;
}

template<typename T>
__device__ static inline
T min(T const& a, T const& b){
    return (a < b) ? a : b;
}


//passing by copy is important!
__global__ static
void compute_partial_centroids_kernel(
    DeviceVector<Sample> const sv,
    DeviceVector<Sample> const centroids,
    DeviceVector<Sample> const centroids_accumulator,
    DeviceVector<size_t> const cluster_sizes_accumulator)
{

    //if thread_uid >= sv.size, it doesn't enter the loop simply

    unsigned const thread_uid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x; //the number of threads

    for(size_t i = thread_uid; i < sv.size; i += stride){

        Sample const& s    = sv.data[i];
        float min_dist     = MAX_FLOAT_VALUE;
        size_t new_cluster = 0;

        for(size_t j = 0; j < centroids.size; ++j){

            float const tmp_dist = distance_sample(s, centroids.data[j]);

            new_cluster = (tmp_dist < min_dist) ? j : new_cluster;
            min_dist    = min(tmp_dist, min_dist);
        }


        size_t const accumulator_ind = thread_uid * centroids.size + new_cluster;
        centroids_accumulator.data[accumulator_ind]     += s;
        cluster_sizes_accumulator.data[accumulator_ind] += 1;
    }
}

//passing by copy is important!
__global__ static
void reduce_centroids_kernel(
    DeviceVector<Sample> const centroids,
    DeviceVector<size_t> const cluster_sizes,
    DeviceVector<Sample> const centroids_accumulator,
    DeviceVector<size_t> const cluster_sizes_accumulator)
{

    unsigned const thread_uid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_uid >= centroids.size)
        return;


    centroids.data[thread_uid] = { 0.f, 0.f };
    cluster_sizes.data[thread_uid] = 0;

    for(size_t i = thread_uid; i < centroids_accumulator.size; i += centroids.size){

        centroids.data[thread_uid] += centroids_accumulator.data[i];
        cluster_sizes.data[thread_uid] += cluster_sizes_accumulator.data[i];

        centroids_accumulator.data[i] = { 0.f, 0.f };
        cluster_sizes_accumulator.data[i] = 0;
    }

    centroids.data[thread_uid] /= cluster_sizes.data[thread_uid];
}


void set_seed(unsigned int const seed){
    std::srand(seed);
}

void kmeans(
    size_t const number_of_samples,
    size_t const number_of_clusters,
    size_t const number_of_blocks_per_grid,
    size_t const number_of_threads_per_block)
{

    DeviceVector<Sample> const sv { number_of_samples };
    fill_samples_vector(sv);

    DeviceVector<Sample> const centroids { number_of_clusters };
    init_clusters_vector(centroids, sv);

    DeviceVector<size_t> const cluster_sizes { number_of_clusters };

    DeviceVector<Sample> const centroids_accumulator {
        number_of_clusters *
        number_of_blocks_per_grid *
        number_of_threads_per_block
    };
    centroids_accumulator.reset();

    DeviceVector<size_t> const cluster_sizes_accumulator {
        number_of_clusters *
        number_of_blocks_per_grid *
        number_of_threads_per_block
    };
    centroids_accumulator.reset();


    for(size_t i = 0; i < NUMBER_OF_ITERATIONS; ++i){

        compute_partial_centroids_kernel
            <<<
                number_of_blocks_per_grid, number_of_threads_per_block
            >>>
            (
                sv, centroids, centroids_accumulator, cluster_sizes_accumulator
            );

        reduce_centroids_kernel
            <<<
                1, number_of_clusters
            >>>
            (
                centroids, cluster_sizes, centroids_accumulator, cluster_sizes_accumulator
            );
    }


    std::unique_ptr<Sample[]> const final_centroids = std::make_unique<Sample[]>(centroids.size);
    cudaMemcpy(
        final_centroids.get(),
        centroids.data,
        sizeof *centroids.data * centroids.size,
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    );

    std::unique_ptr<size_t[]> const final_cluster_sizes =
        std::make_unique<size_t[]>(cluster_sizes.size);
    cudaMemcpy(
        final_cluster_sizes.get(),
        cluster_sizes.data,
        sizeof *cluster_sizes.data * cluster_sizes.size,
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    );

    for(size_t i = 0; i < centroids.size; ++i){

        Sample const centroid = final_centroids[i];
        size_t const size = final_cluster_sizes[i];

        std::printf("Center: (%.3f, %.3f) : Size: %lu\n", centroid.x, centroid.y, size);
    }

    std::printf("Iterations: %lu\n", NUMBER_OF_ITERATIONS);
}

}
