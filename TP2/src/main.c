#include "k_means.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


static unsigned int const SEED = 10;


int main(int const argc, char const* const* const argv){

	size_t params[3] = { 0 };

	if(argc < 3 || argc > 4){
		fprintf(stderr, "usage: %s <SAMPLES> <CLUSTERS> [THREADS]\n", argv[0]);
		return 2;
	}


	for(int i = 1; i < argc; ++i){

		long const param = atol(argv[i]);

		if(param > 0)
			params[i - 1] = (size_t) param;

		else {
			fprintf(stderr, "%s: invalid argument '%s': Not a positive integer\n", argv[0], argv[i]);
			return 1;
		}
	}

	if(argc == 3){
		params[2] = 1;
		fprintf(stderr, "\033[1m%s: \033[36mnote:\033[0m number of threads not specified; assuming 1 thread\n\n", argv[0]);
	}


	set_seed(SEED);

	kmeans(params[0], params[1], params[2]);

	return 0;
}
