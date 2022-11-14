#!/usr/bin/env perl

use warnings;
use strict;

use constant MAX_NUM_OF_THREADS => 8;
use constant MAX_NUM_OF_CPUS => 8;
use constant NUM_OF_ITERS => 5;


sub main(){
	
	unless(scalar(@ARGV) == 1){
		die "usage: $0 <CLUSTERS>\n";
	}

	my $NUM_OF_CLS = $ARGV[0];


	my $DIR = "times${NUM_OF_CLS}clusters";
	mkdir $DIR or die "$!";


	for(my $cpus = 2; $cpus <= MAX_NUM_OF_CPUS; $cpus *= 2){

		for(my $thr = 1; $thr <= MAX_NUM_OF_THREADS; $thr *= 2){

			my $fn = "${DIR}/${cpus}cpus${thr}threads.txt";

			for(my $i = 0; $i < NUM_OF_ITERS; ++$i){
				system "{ srun --partition=cpar --cpus-per-task=$cpus perf stat -e instructions,cycles -M cpi bin/k_means 10000000 $NUM_OF_CLS $thr; } 2>> $fn";
			}
		}
	}
}

main();
