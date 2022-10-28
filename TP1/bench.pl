#!/usr/bin/env perl

use warnings;
use strict;

my $FN = "times.txt";
my $NUM_OF_ITER = 5;

unlink "$FN";

for(my $i = 0; $i < $NUM_OF_ITER; ++$i){
	system "{ srun --partition=cpar perf stat -e instructions,cycles -M cpi make run; } 2>> $FN";
}
