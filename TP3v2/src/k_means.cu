#include "k_means.hpp"

#include <cstddef>
#include <cstdio>
#include <new>
#include <limits>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



namespace kmeans_cuda {

static size_t constexpr NUMBER_OF_THREADS_PER_BLOCK = 32 * 8;
static size_t constexpr NUMBER_OF_BLOCKS_PER_GRID   = 4;
static size_t constexpr NUMBER_OF_ITERATIONS        = 20;
static float  constexpr MAX_FLOAT_VALUE             = std::numeric_limits<float>::max();


// Samples

struct Sample {

    float x, y;

    Sample() : x{0.f}, y {0.f} {}

    __device__
    Sample(float const x, float const y) : x{x}, y{y} {}
};

__device__
Sample const& operator+=(Sample& a, Sample const& b){
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__
Sample const& operator/=(Sample& a, size_t const s){
    a.x /= s;
    a.y /= s;
    return a;
}


// Clusters

struct Cluster {

    Sample centroid;
    size_t size;

    Cluster() : centroid{}, size{0} {}
};


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
        sizeof samples[0] * sv.size,
        cudaMemcpyKind::cudaMemcpyHostToDevice
    );
}

static inline
void init_clusters_vector(DeviceVector<Cluster> const &cv, DeviceVector<Sample> const &sv){

    for(size_t i = 0; i < cv.size; ++i)
        cudaMemcpy(
            &cv.data[i].centroid,
            sv.data + i,
            sizeof cv.data[i].centroid,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice
        );
}


static inline
void reset_clusters_vector(DeviceVector<Cluster> const &cv){
    cudaMemset(cv.data, 0, sizeof *cv.data * cv.size);
}

__device__ static inline
float distance_sample(Sample const s1, Sample const s2){

    float const x_diff = s1.x - s2.x;
    float const y_diff = s1.y - s2.y;

    return x_diff * x_diff + y_diff * y_diff;
}

template<typename T>
__device__ static inline
T const& min(T const& a, T const& b){
    return (a < b) ? a : b;
}


//passing by copy is important!
__global__ static
void accumulate_kernel(
    DeviceVector<Sample>  const sv,
    DeviceVector<Cluster> const cv,
    DeviceVector<Cluster> const accumulator)
{

    unsigned const thread_uid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_uid >= sv.size)
        return;


    size_t const stride = gridDim.x * blockDim.x;

    for(size_t i = thread_uid; i < sv.size; i += stride){

        Sample const& s    = sv.data[i];

        float min_dist     = MAX_FLOAT_VALUE;
        size_t new_cluster = 0;

        for(size_t j = 0; j < cv.size; ++j){

            float const tmp_dist = distance_sample(s, cv.data[j].centroid );

            new_cluster = (tmp_dist < min_dist) ? j : new_cluster;
            min_dist    = min(tmp_dist, min_dist);
        }


        size_t const accumulator_ind = thread_uid * cv.size + new_cluster;
        accumulator.data[accumulator_ind].centroid += s;
        accumulator.data[accumulator_ind].size += 1;
    }
}

//passing by copy is important!
__global__ static
void update_clusters_kernel(
    DeviceVector<Cluster> const cv,
    DeviceVector<Cluster> const accumulator)
{

    unsigned const thread_uid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_uid >= cv.size)
        return;


    cv.data[thread_uid].centroid = { 0.f, 0.f };
    cv.data[thread_uid].size = 0;

    for(size_t i = thread_uid; i < accumulator.size; i += cv.size){

        cv.data[thread_uid].centroid += accumulator.data[i].centroid;
        cv.data[thread_uid].size += accumulator.data[i].size;

        accumulator.data[i].centroid = { 0.f, 0.f };
        accumulator.data[i].size = 0;
    }

    cv.data[thread_uid].centroid /= cv.data[thread_uid].size;
}


void set_seed(unsigned int const seed){
    std::srand(seed);
}

void kmeans(size_t const number_of_samples, size_t const number_of_clusters){

    DeviceVector<Sample> const sv { number_of_samples };
    fill_samples_vector(sv);

    DeviceVector<Cluster> const cv { number_of_clusters };
    init_clusters_vector(cv, sv);

    DeviceVector<Cluster> const accumulator {
        number_of_clusters *
        NUMBER_OF_BLOCKS_PER_GRID *
        NUMBER_OF_THREADS_PER_BLOCK
    };
    reset_clusters_vector(accumulator);

    for(size_t i = 0; i < NUMBER_OF_ITERATIONS; ++i){

        accumulate_kernel
            <<<
                NUMBER_OF_BLOCKS_PER_GRID, NUMBER_OF_THREADS_PER_BLOCK
            >>>
            (
                sv, cv, accumulator
            );

        update_clusters_kernel
            <<<
                1, number_of_clusters
            >>>
            (
                cv, accumulator
            );
    }


    std::unique_ptr<Cluster[]> const final_clusters = std::make_unique<Cluster[]>(cv.size);
    cudaMemcpy(
        final_clusters.get(),
        cv.data,
        sizeof *cv.data * cv.size,
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    );

    for(size_t i = 0; i < cv.size; ++i){

        Sample const centroid = final_clusters[i].centroid;
        size_t const size = final_clusters[i].size;

        std::printf("Center: (%.3f, %.3f) : Size: %lu\n", centroid.x, centroid.y, size);
    }

    std::printf("Iterations: %lu\n", NUMBER_OF_ITERATIONS);
}

}
