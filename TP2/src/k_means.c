#include "k_means.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <memory.h>



//Samples

typedef struct {

	float x, y;

} Sample;

typedef struct {

	float x, y;
	long tag;

} TaggedSample;

typedef struct {

	TaggedSample* const restrict data;
	size_t const size;

} TaggedSampleVector;


static inline
TaggedSampleVector new_tagged_sample_vector(size_t const size){

	TaggedSampleVector const tsv = (TaggedSampleVector){
		.data = malloc(sizeof *(tsv.data) * size),
		.size = size
	};

	return tsv;
}

static inline
void fill_tagged_sample_vector(TaggedSampleVector const* const restrict tsv){

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
void delete_tagged_sample_vector(TaggedSampleVector const tsv){
	free(tsv.data);
}



// Clusters

typedef struct {

	Sample centroid;
	size_t size;

} Cluster;

typedef struct {

	Cluster* restrict data;
	size_t const size;

} ClusterVector;


static inline
ClusterVector new_cluster_vector(size_t const NUMBER_OF_CLUSTERS) {

	ClusterVector const cv = (ClusterVector){
		.data = malloc(sizeof *(cv.data) * NUMBER_OF_CLUSTERS),
		.size = NUMBER_OF_CLUSTERS
	};

	return cv;
}

static inline
void init_cluster_vector(ClusterVector const* const restrict cv, TaggedSampleVector const* const restrict tsv){

	for(size_t i = 0; i < cv->size; ++i){

		float const x = tsv->data[i].x;
		float const y = tsv->data[i].y;

		cv->data[i].centroid = (Sample){ .x = x, .y = y };
	}
}

static inline
void swap_data_cluster_vector(ClusterVector* const restrict p1, ClusterVector* const restrict p2){
	Cluster* const tmp = p1->data;
	p1->data = p2->data;
	p2->data = tmp;
}

static inline
void delete_cluster_vector(ClusterVector const cv){
	free(cv.data);
}

static inline
void reset_cluster_vector(ClusterVector const* const restrict cv){
	memset(cv->data, 0, sizeof *(cv->data) * cv->size);
}



void set_seed(unsigned int const seed){
	srand(seed);
}

void kmeans(size_t const NUMBER_OF_SAMPLES, size_t const NUMBER_OF_CLUSTERS){

	TaggedSampleVector const tsv = new_tagged_sample_vector(NUMBER_OF_SAMPLES);
	fill_tagged_sample_vector(&tsv);

	ClusterVector curr_cv = new_cluster_vector(NUMBER_OF_CLUSTERS);
	init_cluster_vector(&curr_cv, &tsv);

	ClusterVector next_cv = new_cluster_vector(NUMBER_OF_CLUSTERS);
	reset_cluster_vector(&next_cv);


	size_t iter = 0;

	for(bool converged = false; !converged; ++iter){

		converged = true;

		for(size_t i = 0; i < tsv.size; ++i){

			Sample const s    = (Sample){ .x = tsv.data[i].x, .y = tsv.data[i].y };
			long  new_cluster = 0;
			float min_dist    = distance_sample(curr_cv.data[0].centroid, s);

			for(size_t j = 1; j < curr_cv.size; ++j){

				float const tmp_dist = distance_sample(curr_cv.data[j].centroid, s);

				new_cluster = (tmp_dist < min_dist) ? (long) j : new_cluster;
				min_dist    = (tmp_dist < min_dist) ? tmp_dist : min_dist;
			}

			converged       = tsv.data[i].tag == new_cluster && converged;
			tsv.data[i].tag = tsv.data[i].tag == new_cluster ? tsv.data[i].tag : new_cluster;

			next_cv.data[new_cluster].centroid.x += s.x;
			next_cv.data[new_cluster].centroid.y += s.y;
			next_cv.data[new_cluster].size += 1;
		}

	    /* When converged == true (final iteration),
		 * code below this comment is redundant.
		 * But at least there are no if clauses...
   		 */

		for(size_t i = 0; i < next_cv.size; ++i){
			next_cv.data[i].centroid.x /= next_cv.data[i].size;
			next_cv.data[i].centroid.y /= next_cv.data[i].size;
		}

		swap_data_cluster_vector(&curr_cv, &next_cv);
		reset_cluster_vector(&next_cv);
	}

	/* Final iteration is merely a verification.
	 * Thus, we need to decrement iter.
	 */

	--iter;

	for(size_t i = 0; i < curr_cv.size; ++i){

		Sample const centroid = curr_cv.data[i].centroid;

		printf("Center: (%.3f, %.3f) : Size: %lu\n", centroid.x, centroid.y, curr_cv.data[i].size);
	}

	printf("Iterations: %lu\n", iter);

	delete_cluster_vector(curr_cv);
	delete_cluster_vector(next_cv);
	delete_tagged_sample_vector(tsv);
}
