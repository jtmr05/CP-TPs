#include "k_means.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


static unsigned int const SEED = 10;

static char const* const REGEX_COMMAND = "perl -e 'exit (($ARGV[0] =~ m/^\\+?(\\d+)$/) ? ($1 == 0) : 1);'";


int main(int const argc, char const* const* const argv){

	size_t params[2] = { 0 };

	if(argc != 3){
		fprintf(stderr, "usage: %s <SAMPLES> <CLUSTERS>\n", argv[0]);
		return 2;
	}


	for(int i = 1; i < argc; ++i){

		size_t const size = strlen(REGEX_COMMAND) + strlen(argv[i]) + 2;
		char* const command = (char*) malloc(size);

		snprintf(command, size, "%s %s", REGEX_COMMAND, argv[i]);

		if(system(command) == 0)
			params[i - 1] = (size_t) atol(argv[i]);

		else {

			fprintf(stderr, "%s: invalid argument '%s' (must be a positive integer)\n", argv[0], argv[i]);
			free(command);
			return 1;
		}

		free(command);
	}


	set_seed(SEED);

	kmeans(params[0], params[1]);

	return 0;
}
