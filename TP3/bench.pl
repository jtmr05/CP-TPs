#!/usr/bin/env perl

use warnings;
use strict;

use constant CLUSTERS   => [4, 20, 32];
use constant SAMPLES    => [1000];
use constant THREAD_LIM => 1024;
use constant BLOCKS		=> 32;
use constant DIR        => 'benchmarks';


sub main(){
	
	unless(-d DIR){
		mkdir DIR or die "$!";
	}


	foreach my $s (@{SAMPLES()}){

		foreach my $c (@{CLUSTERS()}){

			my $fn = sprintf '%s/%dsamples%dclusters.out', DIR, $s, $c;

			for(my $t = 1; $t <= THREAD_LIM; $t *= 2){
				
				my $command = sprintf
					'hyperfine -M 5 "bin/k_means %d %d %d %d" >> %s 2> /dev/null',
					$s, $c, BLOCKS, $t, $fn;

				system $command;
			}
		}
	}
}


main();
