#include "k_means.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

static const size_t NUMBER_OF_THREADS = 4;
static const size_t NUMBER_OF_BLOCKS = 1;
static size_t const MAX_ITERS = 20;

// Samples

typedef struct
{

    float x, y;

} Sample;

typedef struct
{

    float x, y;
    long tag;

} TaggedSample;

typedef struct
{

    TaggedSample *data;
    size_t const size;

} TaggedSampleVector;

static inline TaggedSampleVector new_tagged_sample_vector(size_t const size)
{

    TaggedSampleVector tsv = (TaggedSampleVector){
        .data = NULL,
        .size = size};

    cudaMalloc((void **)&tsv.data, sizeof *(tsv.data) * size);

    return tsv;
}

static inline void fill_tagged_sample_vector(TaggedSampleVector const *const tsv)
{

    for (size_t i = 0; i < tsv->size; ++i)
    {

        float const x = (float)rand() / (float)RAND_MAX;
        float const y = (float)rand() / (float)RAND_MAX;

        TaggedSample ts = (TaggedSample){.x = x, .y = y, .tag = -1};
        cudaMemcpy(&tsv->data[i], &ts, sizeof(ts), cudaMemcpyHostToDevice);
    }
}

__device__ static inline float distance_sample(Sample const s1, Sample const s2)
{

    float const x_diff = s1.x - s2.x;
    float const y_diff = s1.y - s2.y;

    return x_diff * x_diff + y_diff * y_diff;
}

static inline void delete_tagged_sample_vector(TaggedSampleVector const *const tsv)
{
    cudaFree(tsv->data);
}

__device__ static inline void swap_pointers(void **p1, void **p2)
{

    void *tmp = *p1;

    *p1 = *p2;
    *p2 = tmp;
}

__device__ static inline void reset(float *p, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        p[i] = 0;
    }
}

__device__ static inline void reset(size_t *p, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        p[i] = 0;
    }
}

void set_seed(unsigned int const seed)
{
    srand(seed);
}

__global__ void kmeansKernel(TaggedSampleVector tsv, size_t cv_size)
{
    size_t const size = (tsv.size / NUMBER_OF_THREADS) + 1;

    extern __shared__ float shared_memory[];
    float *curr_xs = shared_memory;
    float *curr_ys = shared_memory + cv_size;
    size_t *curr_sizes = (size_t *)shared_memory + cv_size * 2;
    float *next_xs = shared_memory + cv_size * 3;
    float *next_ys = shared_memory + cv_size * 4;
    size_t *next_sizes = (size_t *)shared_memory + cv_size * 5;

    if (threadIdx.x == 0)
    {

        for (size_t i = 0; i < cv_size; ++i)
        {
            curr_sizes[i] = 0;
            curr_xs[i] = tsv.data[i].x;
            curr_ys[i] = tsv.data[i].y;
        }

        reset(next_xs, sizeof *next_xs * cv_size);
        reset(next_ys, sizeof *next_ys * cv_size);
        reset(next_sizes, sizeof *next_sizes * cv_size);
    }

    __syncthreads();

    for (size_t iter = 0; iter < MAX_ITERS; ++iter)
    {

        for (size_t i = threadIdx.x * size; i < (threadIdx.x + 1) * size && i < tsv.size; ++i)
        {

            Sample const s = (Sample){.x = tsv.data[i].x, .y = tsv.data[i].y};
            Sample centroid = (Sample){.x = curr_xs[0], .y = curr_ys[0]};
            float min_dist = distance_sample(s, centroid);

            long new_cluster = 0;

            for (size_t j = 1; j < cv_size; ++j)
            {

                centroid = (Sample){.x = curr_xs[j], .y = curr_ys[j]};
                float const tmp_dist = distance_sample(s, centroid);

                new_cluster = (tmp_dist < min_dist) ? (long)j : new_cluster;
                min_dist = (tmp_dist < min_dist) ? tmp_dist : min_dist;
            }

            tsv.data[i].tag = new_cluster;

            atomicAdd(&curr_xs[new_cluster], s.x);
            atomicAdd(&curr_ys[new_cluster], s.y);
            atomicAdd((unsigned long long *)&curr_sizes[new_cluster], (unsigned long long)1);
        }

        if (threadIdx.x == 0)
        {
            for (size_t i = 0; i < cv_size; ++i)
            {
                next_xs[i] /= next_sizes[i];
                next_ys[i] /= next_sizes[i];
            }
            swap_pointers((void **)&curr_xs, (void **)&next_xs);
            swap_pointers((void **)&curr_ys, (void **)&next_ys);
            swap_pointers((void **)&curr_sizes, (void **)&next_sizes);

            reset(next_xs, sizeof *next_xs * cv_size);
            reset(next_ys, sizeof *next_ys * cv_size);
            reset(next_sizes, sizeof *next_sizes * cv_size);
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (size_t i = 0; i < cv_size; ++i)
        {

            float const x = curr_xs[i];
            float const y = curr_ys[i];
            size_t const size = curr_sizes[i];

            printf("Center: (%.3f, %.3f) : Size: %lu\n", x, y, size);
        }

        printf("Iterations: %lu\n", MAX_ITERS);
    }

}

void kmeans(size_t const NUMBER_OF_SAMPLES, size_t const NUMBER_OF_CLUSTERS)
{

    TaggedSampleVector const tsv = new_tagged_sample_vector(NUMBER_OF_SAMPLES);
    fill_tagged_sample_vector(&tsv);

    kmeansKernel<<< NUMBER_OF_BLOCKS,NUMBER_OF_THREADS, (4 * sizeof(float) + 2 * sizeof(size_t)) * NUMBER_OF_CLUSTERS>>>(tsv, NUMBER_OF_CLUSTERS);

    delete_tagged_sample_vector(&tsv);
}
