#include "k_means.hpp"

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

    float const x, y;
    long tag;

    TaggedSample(float const x, float const y, long const tag) : x(x), y(y), tag(tag) {}
};

struct TaggedSampleVector {

    TaggedSample* data;
    size_t const size;

    TaggedSampleVector(size_t const size) : data(nullptr), size(size) {
        TaggedSample* ptr = nullptr;
        cudaMalloc(&ptr, sizeof *(this->data) * size);
        this->data = ptr;
    }

    ~TaggedSampleVector(){
        cudaFree(this->data);
    }

    void fill() const {

        for (size_t i = 0; i < this->size; ++i){

            float const x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            float const y = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            TaggedSample const ts { x, y, -1};
            cudaMemcpy(this->data + i, &ts, sizeof(ts), cudaMemcpyHostToDevice);
        }
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
void reset(T* const p, size_t const size){

    for (size_t i = 0; i < size; i++){
        p[i] = static_cast<T>(0);
    }
}

void set_seed(unsigned int const seed){
    std::srand(seed);
}

template<typename T>
__device__ static inline
T* place_into_byte_arr(std::byte* const base, size_t const size){

    static size_t offset = 0;
    T* const obj = new(base + offset) T[size];
    offset += sizeof *obj * size;
    return obj;
}

__global__ static 
void kmeans_kernel(TaggedSampleVector const& tsv, size_t const NUMBER_OF_CLUSTERS){

    extern __shared__ std::byte shared_memory[];

    float* curr_xs = nullptr; float* curr_ys = nullptr; lsize_t* curr_sizes = nullptr;
    float* next_xs = nullptr; float* next_ys = nullptr; lsize_t* next_sizes = nullptr;


    if (threadIdx.x == 0){

        curr_xs    = place_into_byte_arr<float>(shared_memory,   NUMBER_OF_CLUSTERS);
        curr_ys    = place_into_byte_arr<float>(shared_memory,   NUMBER_OF_CLUSTERS);
        curr_sizes = place_into_byte_arr<lsize_t>(shared_memory, NUMBER_OF_CLUSTERS);

        next_xs    = place_into_byte_arr<float>(shared_memory,   NUMBER_OF_CLUSTERS);
        next_ys    = place_into_byte_arr<float>(shared_memory,   NUMBER_OF_CLUSTERS);
        next_sizes = place_into_byte_arr<lsize_t>(shared_memory, NUMBER_OF_CLUSTERS);

        for (size_t i = 0; i < NUMBER_OF_CLUSTERS; ++i){
            curr_xs[i] = tsv.data[i].x;
            curr_ys[i] = tsv.data[i].y;
            curr_sizes[i] = 0;
        }

        reset(next_xs, NUMBER_OF_CLUSTERS);
        reset(next_ys, NUMBER_OF_CLUSTERS);
        reset(next_sizes, NUMBER_OF_CLUSTERS);
    }

    __syncthreads();


    size_t const local_size = (tsv.size / NUMBER_OF_THREADS_PER_BLOCK) + 1;

    for (size_t iter = 0; iter < NUMBER_OF_ITERATIONS; ++iter){

        for (size_t i = threadIdx.x * local_size; i < (threadIdx.x + 1) * local_size && i < tsv.size; ++i){

            Sample const s  { tsv.data[i].x, tsv.data[i].y };
            Sample centroid { curr_xs[0], curr_ys[0] };

            float min_dist   = distance_sample(s, centroid);
            long new_cluster = 0;

            for (size_t j = 1; j < NUMBER_OF_CLUSTERS; ++j){

                centroid = { curr_xs[j], curr_ys[j] };
                float const tmp_dist = distance_sample(s, centroid);

                new_cluster = (tmp_dist < min_dist) ? static_cast<long>(j) : new_cluster;
                min_dist    = (tmp_dist < min_dist) ? tmp_dist : min_dist;
            }

            tsv.data[i].tag = new_cluster;

            atomicAdd(curr_xs + new_cluster, s.x);
            atomicAdd(curr_ys + new_cluster, s.y);
            atomicAdd(curr_sizes + new_cluster, 1ULL);
        }

        if (threadIdx.x == 0){

            for (size_t i = 0; i < NUMBER_OF_CLUSTERS; ++i){
                next_xs[i] /= next_sizes[i];
                next_ys[i] /= next_sizes[i];
            }
            
            swap_pointers(curr_xs, next_xs);
            swap_pointers(curr_ys, next_ys);
            swap_pointers(curr_sizes, next_sizes);

            reset(next_xs, NUMBER_OF_CLUSTERS);
            reset(next_ys, NUMBER_OF_CLUSTERS);
            reset(next_sizes, NUMBER_OF_CLUSTERS);
        }

        __syncthreads();
    }


    if (threadIdx.x == 0){

        for (size_t i = 0; i < NUMBER_OF_CLUSTERS; ++i){

            float const x = curr_xs[i];
            float const y = curr_ys[i];
            lsize_t const size = curr_sizes[i];

            std::printf("Center: (%.3f, %.3f) : Size: %llu\n", x, y, size);
        }

        std::printf("Iterations: %lu\n", NUMBER_OF_ITERATIONS);
    }
}

void kmeans(size_t const NUMBER_OF_SAMPLES, size_t const NUMBER_OF_CLUSTERS){

    TaggedSampleVector const tsv { NUMBER_OF_SAMPLES };
    tsv.fill();

    size_t const allocated_shared_memory = (4 * sizeof(float) + 2 * sizeof(lsize_t)) * NUMBER_OF_CLUSTERS;

    kmeans_kernel<<<NUMBER_OF_BLOCKS_PER_GRID, NUMBER_OF_THREADS_PER_BLOCK, allocated_shared_memory>>>(tsv, NUMBER_OF_CLUSTERS);
}

}
