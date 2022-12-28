#include "k_means.hpp"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <array>


static unsigned int constexpr SEED = 10;


static inline
std::optional<size_t> from_string(char const* const c_str){
	
	long const value = std::atol(c_str);

	if(value > 0)
		return std::make_optional(static_cast<size_t>(value));

	return std::nullopt;
}

int main(int const argc, char const* const* const argv){

	std::array<size_t, 2> params;

	if(argc != 3){
		std::fprintf(stderr, "usage: %s <SAMPLES> <CLUSTERS>\n", argv[0]);
		return 2;
	}


	for(size_t i = 1; i < static_cast<size_t>(argc); ++i){

		auto const res = from_string(argv[i]);

		if(res.has_value())
			params.at(i - 1) = res.value();

		else {
			std::fprintf(stderr, "%s: invalid argument '%s': Not a positive integer\n", argv[0], argv[i]);
			return 1;
		}
	}


	kmeans_cuda::set_seed(SEED);
	kmeans_cuda::kmeans(params.at(0), params.at(1));

	return 0;
}
