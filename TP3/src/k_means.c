#include "k_means.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



static const size_t NUMBER_OF_THREADS = 1024;
static const size_t NUMBER_OF_BLOCKS = 1;
static size_t const MAX_ITERS = 20;



//Samples

typedef struct {

    float x, y;

} Sample;

typedef struct {

    float x, y;
    long tag;

} TaggedSample;

typedef struct {

    TaggedSample* data;
    size_t const size;

} TaggedSampleVector;


static inline
TaggedSampleVector new_tagged_sample_vector(size_t const size){

    TaggedSampleVector tsv = (TaggedSampleVector){
        .data = NULL,
        .size = size
    };

    cudaMalloc(&tsv.data, sizeof *(tsv.data) * size);

    return tsv;
}

static inline
void fill_tagged_sample_vector(TaggedSampleVector const* const tsv){

    for(size_t i = 0; i < tsv->size; ++i){

        float const x = (float) rand() / (float) RAND_MAX;
        float const y = (float) rand() / (float) RAND_MAX;

        tsv->data[i] = (TaggedSample){ .x = x, .y = y, .tag = -1 };
    }
}

static inline
float distance_sample(Sample const s1, Sample const s2){

    float const x_diff = s1.x - s2.x;
    float const y_diff = s1.y - s2.y;

    return x_diff * x_diff + y_diff * y_diff;
}

static inline
void delete_tagged_sample_vector(TaggedSampleVector const* const tsv){
    cudaFree(tsv->data);
}



// Clusters

typedef struct {

    __shared__ float* xs;
    __shared__ float* ys;
    __shared__ size_t* sizes;
    size_t const size;

} ClusterVector;

static inline
ClusterVector new_cluster_vector(size_t const NUMBER_OF_CLUSTERS) {

    ClusterVector cv = (ClusterVector){
        .xs = NULL,
        .ys = NULL,
        .sizes = NULL,
        .size = NUMBER_OF_CLUSTERS
    };

    cudaMalloc(&cv.xs, sizeof *(cv.xs) * NUMBER_OF_CLUSTERS);
    cudaMalloc(&cv.ys, sizeof *(cv.ys) * NUMBER_OF_CLUSTERS);
    cudaMalloc(&cv.sizes, sizeof *(cv.sizes) * NUMBER_OF_CLUSTERS);

    return cv;
}

static inline
void init_cluster_vector(ClusterVector const* const cv, TaggedSampleVector const* const tsv){

    for(size_t i = 0; i < cv->size; ++i){

        cv->xs[i] = tsv->data[i].x;
        cv->ys[i] = tsv->data[i].y;
    }
}

static inline
void swap_data_cluster_vector(ClusterVector* const p1, ClusterVector* const p2){

    float* const tmp_xs = p1->xs;
    float* const tmp_ys = p1->ys;
    size_t* const tmp_sizes = p1->sizes;

    p1->xs = p2->xs;
    p1->ys = p2->ys;
    p1->sizes = p2->sizes;

    p2->xs = tmp_xs;
    p2->ys = tmp_ys;
    p2->sizes = tmp_sizes;
}

static inline
void delete_cluster_vector(ClusterVector const* const cv){
    free(cv->xs);
    free(cv->ys);
    free(cv->sizes);
}

static inline
void reset_cluster_vector(ClusterVector const* const cv){
    memset(cv->xs, 0, sizeof *(cv->xs) * cv->size);
    memset(cv->ys, 0, sizeof *(cv->ys) * cv->size);
    memset(cv->sizes, 0, sizeof *(cv->sizes) * cv->size);
}



void set_seed(unsigned int const seed){
    srand(seed);
}

__global__
void kmeansKernel (TaggedSampleVector tsv,size_t NUMBER_OF_CLUSTERS ){
__shared__ ClusterVector nextcv = new_cluster_vector(NUMBER_OF_CLUSTERS);
int j = threadIdx.x;
size_t size = tsv.size/nTreads;




}
void kmeans(size_t const NUMBER_OF_SAMPLES, size_t const NUMBER_OF_CLUSTERS){

    TaggedSampleVector const tsv = new_tagged_sample_vector(NUMBER_OF_SAMPLES);
    fill_tagged_sample_vector(&tsv);

    ClusterVector curr_cv = new_cluster_vector(NUMBER_OF_CLUSTERS);
    init_cluster_vector(&curr_cv, &tsv);

    ClusterVector next_cv = new_cluster_vector(NUMBER_OF_CLUSTERS);
    reset_cluster_vector(&next_cv);



    for(size_t iter = 0; iter < MAX_ITERS; ++iter){

        // Get the pointers to be used for reduction
        float* const xs = next_cv.xs;
        float* const ys = next_cv.ys;
        size_t* const sizes = next_cv.sizes;

        

        int ThreadsPerBlock= 1024;
        int blocksPerGrid = 1;
        kmeansKernel<<<blocksPerGrid,ThreadsPerBlock>>> (...);

        for(size_t i = 0; i < tsv.size; ++i){

            Sample const s   = (Sample){ .x = tsv.data[i].x, .y = tsv.data[i].y };
            Sample centroid  = (Sample){ .x = curr_cv.xs[0], .y = curr_cv.ys[0] };
            float min_dist   = distance_sample(s, centroid);

            long new_cluster = 0;


            for(size_t j = 1; j < curr_cv.size; ++j){

                centroid = (Sample){ .x = curr_cv.xs[j], .y = curr_cv.ys[j] };
                float const tmp_dist = distance_sample(s, centroid);

                new_cluster = (tmp_dist < min_dist) ? (long) j : new_cluster;
                min_dist    = (tmp_dist < min_dist) ? tmp_dist : min_dist;
            }

            tsv.data[i].tag = new_cluster;

            xs[new_cluster] += s.x;
            ys[new_cluster] += s.y;
            sizes[new_cluster] += 1;
        }


        for(size_t i = 0; i < next_cv.size; ++i){
            next_cv.xs[i] /= next_cv.sizes[i];
            next_cv.ys[i] /= next_cv.sizes[i];
        }

        swap_data_cluster_vector(&curr_cv, &next_cv);
        reset_cluster_vector(&next_cv);
    }


    for(size_t i = 0; i < curr_cv.size; ++i){

        float const x = curr_cv.xs[i];
        float const y = curr_cv.ys[i];
        size_t const size = curr_cv.sizes[i];

        printf("Center: (%.3f, %.3f) : Size: %lu\n", x, y, size);
    }

    printf("Iterations: %lu\n", MAX_ITERS);

    delete_cluster_vector(&curr_cv);
    delete_cluster_vector(&next_cv);
    delete_tagged_sample_vector(&tsv);
}
