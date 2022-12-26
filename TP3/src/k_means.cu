#include "k_means.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <new>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>


namespace kmeans_cuda {

static size_t constexpr NUMBER_OF_THREADS_PER_BLOCK = 4;
static size_t constexpr NUMBER_OF_BLOCKS_PER_GRID   = 1;
static size_t constexpr NUMBER_OF_ITERATIONS        = 20;

// long size_t
// needed because CUDA doesn't suport atomic operations on size_t
typedef unsigned long long lsize_t;


// Samples

struct Sample {

    float x, y;

    __device__
    Sample(float const x, float const y) : x(x), y(y) {}
};

struct TaggedSample {

    float x, y;
    long tag;

    TaggedSample() = default;
    TaggedSample(float const x, float const y, long const tag) : x(x), y(y), tag(tag) {}
};

struct DeviceTaggedSampleVector {

    TaggedSample* data;
    size_t const size;

    DeviceTaggedSampleVector(TaggedSample* const data, size_t const size) : data(data), size(size) {}
};

struct HostTaggedSampleVector {

    TaggedSample* data;
    size_t const size;

    HostTaggedSampleVector(size_t const size) : data(new TaggedSample[size]), size(size) {}

    ~HostTaggedSampleVector(){
        delete[] this->data;
    }

    void fill() const {

        for(size_t i = 0; i < this->size; ++i){

            float const x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            float const y = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            this->data[i] = { x, y, -1 };
        }
    }
};


class MemoryPool {

private:

    std::byte* const begin;
    size_t const size;    //for the sake of completeness
    size_t offset;

public:

    __device__
    MemoryPool(std::byte* const buffer, size_t const buffer_size) :
        begin(buffer),
        size(buffer_size),
        offset(0) {}

    template<typename T>
    __device__
    T* get(size_t const nmemb){

        size_t const requested_bytes = sizeof(T) * nmemb;

        if(this->size < this->offset + requested_bytes)
            return nullptr;

        T* const obj = new(this->begin + this->offset) T[nmemb];
        this->offset += requested_bytes;
        return obj;
    }
};


__device__ static inline
float distance_sample(Sample const s1, Sample const s2){

    float const x_diff = s1.x - s2.x;
    float const y_diff = s1.y - s2.y;

    return x_diff * x_diff + y_diff * y_diff;
}

template<typename T>
__device__ static inline
void swap_pointers(T*& p1, T*& p2){

    T* const tmp = p1;
    p1 = p2;
    p2 = tmp;
}

template<typename T>
__device__ static inline
void reset(T* const p, size_t const nmemb){

    for(size_t i = 0; i < nmemb; i++){
        p[i] = static_cast<T>(0);
    }
}

void set_seed(unsigned int const seed){
    std::srand(seed);
}


template<typename T>
__device__ static inline
T const& min(T const& a, T const& b){
    return (a < b) ? a : b;
}

__global__ static
void kmeans_kernel(DeviceTaggedSampleVector const* const tsv_ptr, size_t const number_of_clusters, size_t const byte_buffer_size){

    DeviceTaggedSampleVector const& tsv = *tsv_ptr;


    extern __shared__ std::byte byte_buffer[];

    float* curr_xs = nullptr; float* curr_ys = nullptr; lsize_t* curr_sizes = nullptr;
    float* next_xs = nullptr; float* next_ys = nullptr; lsize_t* next_sizes = nullptr;


    if(threadIdx.x == 0){

        MemoryPool mp { byte_buffer, byte_buffer_size };

        curr_xs    = mp.get<float>(number_of_clusters);
        curr_ys    = mp.get<float>(number_of_clusters);
        curr_sizes = mp.get<lsize_t>(number_of_clusters);

        next_xs    = mp.get<float>(number_of_clusters);
        next_ys    = mp.get<float>(number_of_clusters);
        next_sizes = mp.get<lsize_t>(number_of_clusters);


        for(size_t i = 0; i < number_of_clusters; ++i){
            curr_xs[i] = tsv.data[i].x;
            curr_ys[i] = tsv.data[i].y;
            curr_sizes[i] = 0;
        }

        reset(next_xs, number_of_clusters);
        reset(next_ys, number_of_clusters);
        reset(next_sizes, number_of_clusters);
    }


    size_t const chunk_size = (tsv.size / NUMBER_OF_THREADS_PER_BLOCK) + 1;
    size_t const begin      = threadIdx.x * chunk_size;
    size_t const end        = min((threadIdx.x + 1) * chunk_size, tsv.size);

    for(size_t iter = 0; iter < NUMBER_OF_ITERATIONS; ++iter){

        //sync all threads before an iteration
        //ensures current clusters aren't being written
        __syncthreads();

        for(size_t i = begin; i < end; ++i){

            Sample const s  { tsv.data[i].x, tsv.data[i].y };
            Sample centroid { curr_xs[0], curr_ys[0] };

            float min_dist   = distance_sample(s, centroid);
            long new_cluster = 0;

            for(size_t j = 1; j < number_of_clusters; ++j){

                centroid = { curr_xs[j], curr_ys[j] };
                float const tmp_dist = distance_sample(s, centroid);

                new_cluster = (tmp_dist < min_dist) ? static_cast<long>(j) : new_cluster;
                min_dist    = min(tmp_dist, min_dist);
            }

            tsv.data[i].tag = new_cluster;

            atomicAdd(next_xs + new_cluster, s.x);
            atomicAdd(next_ys + new_cluster, s.y);
            atomicAdd(next_sizes + new_cluster, 1ULL);
        }

        //ensure all threads have processed their assigned chunk
        __syncthreads();

        if(threadIdx.x == 0){

            for(size_t i = 0; i < number_of_clusters; ++i){
                next_xs[i] /= next_sizes[i];
                next_ys[i] /= next_sizes[i];
            }

            swap_pointers(curr_xs, next_xs);
            swap_pointers(curr_ys, next_ys);
            swap_pointers(curr_sizes, next_sizes);

            reset(next_xs, number_of_clusters);
            reset(next_ys, number_of_clusters);
            reset(next_sizes, number_of_clusters);
        }
    }


    if(threadIdx.x == 0){

        for(size_t i = 0; i < number_of_clusters; ++i){

            float const x = curr_xs[i];
            float const y = curr_ys[i];
            lsize_t const size = curr_sizes[i];

            std::printf("Center: (%.3f, %.3f) : Size: %llu\n", x, y, size);
        }

        std::printf("Iterations: %lu\n", NUMBER_OF_ITERATIONS);
    }
}

void kmeans(size_t const number_of_samples, size_t const number_of_clusters){

    DeviceTaggedSampleVector* device_tsv = nullptr;

    {
        HostTaggedSampleVector const host_tsv { number_of_samples };
        host_tsv.fill();

        TaggedSample* dtsv_data = nullptr;
        cudaMalloc(&dtsv_data, sizeof *dtsv_data * host_tsv.size);
        cudaMemcpy(dtsv_data, host_tsv.data, sizeof *host_tsv.data * host_tsv.size, cudaMemcpyKind::cudaMemcpyHostToDevice);


        DeviceTaggedSampleVector const tmp_dtsv { dtsv_data, host_tsv.size };

        cudaMalloc(&device_tsv, sizeof *device_tsv);
        cudaMemcpy(device_tsv, &tmp_dtsv, sizeof tmp_dtsv, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }


    size_t const shared_mem_size = (4 * sizeof(float) + 2 * sizeof(lsize_t)) * number_of_clusters;

    kmeans_kernel<<<
        NUMBER_OF_BLOCKS_PER_GRID, NUMBER_OF_THREADS_PER_BLOCK, shared_mem_size
        >>>(
            device_tsv, number_of_clusters, shared_mem_size
    );

    cudaFree(device_tsv->data);
    cudaFree(device_tsv);
}

}
