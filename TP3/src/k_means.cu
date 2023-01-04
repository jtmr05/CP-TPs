#include "k_means.hpp"

#include <cstddef>
#include <cstdio>
#include <new>
#include <limits>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>


namespace kmeans_cuda {

static size_t constexpr NUMBER_OF_THREADS_PER_BLOCK = 32 * 1;
static size_t constexpr NUMBER_OF_BLOCKS_PER_GRID   = 1;
static size_t constexpr NUMBER_OF_ITERATIONS        = 20;
static float  constexpr MAX_FLOAT_VALUE             = std::numeric_limits<float>::max();


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

struct TaggedSampleVector {

    TaggedSample* data;
    size_t const size;
    //true if memory pointed by data was allocated
    //when constructing an object of this class
    bool const is_owner;

    TaggedSampleVector(size_t const size) : data(nullptr), size(size), is_owner(true) {
        cudaMalloc(&(this->data), sizeof *(this->data) * size);
    }

    TaggedSampleVector(TaggedSampleVector const& other) :
        data(other.data),
        size(other.size),
        is_owner(false) {}

    TaggedSampleVector(TaggedSampleVector&& other) = delete;

    ~TaggedSampleVector(){
        if(this->is_owner)
            cudaFree(this->data);
    }

    void fill() const {

        std::unique_ptr<TaggedSample[]> const t_samples =
            std::make_unique<TaggedSample[]>(this->size);

        for (size_t i = 0; i < this->size; ++i){

            float const x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            float const y = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            t_samples[i] = { x, y, -1 };
        }

        cudaMemcpy(
            this->data,
            t_samples.get(),
            sizeof t_samples[0] * this->size,
            cudaMemcpyKind::cudaMemcpyHostToDevice
        );
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

        T* const obj = new(this->begin + this->offset) T[threadIdx.x == 0 ? nmemb : 0];
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
T const& min(T const& a, T const& b){
    return (a < b) ? a : b;
}


//passing tsv by copy is important!
__global__ static
void kmeans_kernel(
    TaggedSampleVector const tsv,
    size_t const number_of_clusters,
    size_t const byte_buffer_size)
{

    extern __shared__ std::byte byte_buffer[];


    MemoryPool mp { byte_buffer, byte_buffer_size };

    auto curr_xs    = mp.get<float>(number_of_clusters);
    auto curr_ys    = mp.get<float>(number_of_clusters);
    auto curr_sizes = mp.get<size_t>(number_of_clusters);

    if(threadIdx.x == 0){

        for(size_t i = 0; i < number_of_clusters; ++i){
            curr_xs[i] = tsv.data[i].x;
            curr_ys[i] = tsv.data[i].y;
            curr_sizes[i] = 0;
        }
    }


    // relative to the (implicit) cluster vector
    size_t const tsv_chunk_size = (tsv.size / NUMBER_OF_THREADS_PER_BLOCK) + 1;
    size_t const begin_tsv_ind  = threadIdx.x * tsv_chunk_size;
    size_t const end_tsv_ind    = min((threadIdx.x + 1) * tsv_chunk_size, tsv.size);

    // relative to the (implicit) cluster vector
    size_t const cv_chunk_size = (number_of_clusters / NUMBER_OF_THREADS_PER_BLOCK) + 1;
    size_t const begin_cv_ind  = threadIdx.x * cv_chunk_size;
    size_t const end_cv_ind    = min((threadIdx.x + 1) * cv_chunk_size, number_of_clusters);


    for(size_t iter = 0; iter < NUMBER_OF_ITERATIONS; ++iter){

        //sync all threads before an iteration
        //ensures current clusters aren't being written
        __syncthreads();

        for(size_t i = begin_tsv_ind; i < end_tsv_ind; ++i){

            Sample const s { tsv.data[i].x, tsv.data[i].y };
            float min_dist   = MAX_FLOAT_VALUE;
            long new_cluster = 0;

            for(size_t j = 0; j < number_of_clusters; ++j){

                Sample const centroid { curr_xs[j], curr_ys[j] };
                float const tmp_dist = distance_sample(s, centroid);

                new_cluster = (tmp_dist < min_dist) ? static_cast<long>(j) : new_cluster;
                min_dist    = min(tmp_dist, min_dist);
            }

            tsv.data[i].tag = new_cluster;
        }


        //ensure all threads have processed their assigned chunk
        __syncthreads();

        for(size_t i = begin_cv_ind; i < end_cv_ind; ++i){
            curr_xs[i] = 0.f;
            curr_ys[i] = 0.f;
            curr_sizes[i] = 0;
        }

        for(size_t i = 0; i < tsv.size; ++i){

            size_t const ind = static_cast<size_t>(tsv.data[i].tag);

            if(begin_cv_ind <= ind && ind < end_cv_ind){
                curr_xs[ind] += tsv.data[i].x;
                curr_ys[ind] += tsv.data[i].y;
                ++curr_sizes[ind];
            }
        }

        for(size_t i = begin_cv_ind; i < end_cv_ind; ++i){
            curr_xs[i] /= curr_sizes[i];
            curr_ys[i] /= curr_sizes[i];
        }
    }


    __syncthreads();

    if(threadIdx.x == 0){

        for(size_t i = 0; i < number_of_clusters; ++i){

            float const x = curr_xs[i];
            float const y = curr_ys[i];
            size_t const size = curr_sizes[i];

            std::printf("Center: (%.3f, %.3f) : Size: %lu\n", x, y, size);
        }

        std::printf("Iterations: %lu\n", NUMBER_OF_ITERATIONS);
    }
}


void set_seed(unsigned int const seed){
    std::srand(seed);
}

void kmeans(size_t const number_of_samples, size_t const number_of_clusters){

    TaggedSampleVector tsv { number_of_samples };
    tsv.fill();

    size_t const shared_mem_size = (2 * sizeof(float) + sizeof(size_t)) * number_of_clusters;

    kmeans_kernel
        <<<
            NUMBER_OF_BLOCKS_PER_GRID, NUMBER_OF_THREADS_PER_BLOCK, shared_mem_size
        >>>
        (
            tsv, number_of_clusters, shared_mem_size
        );
}

}
