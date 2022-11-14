#!/usr/bin/env perl

use strict;
use warnings;

foreach my $fn (@ARGV){

	open my $fh, '<', $fn;

	my %sums = ( 'cycles' => 0, 'instructions' => 0, 'seconds time' => 0 );
	my %counts = ( 'cycles' => 0, 'instructions' => 0, 'seconds time' => 0 );

	while(<$fh>){

		if(m/\s*(.+?)\s+(instructions|cycles|seconds\stime)/){
	
			my $one = $1;
			my $two = $2;

			$one =~ s/,//g;

			$sums{$two} += $one;
			$counts{$two}++;
		}
	}
	

	for my $key (keys %sums){
		$sums{$key} /= $counts{$key};
	}

	print "$fn:\n";

	
	foreach my $key (keys %sums){

		my $rounded = $sums{$key};
		my $decimal = undef;

		$rounded =~ s/(\d+)(\.\d+)?/$1/;
		$decimal = $2;
		
		while($rounded =~ s/(\d+)(\d\d\d)/$1\,$2/){}
		
		print "\t$key => $rounded", $decimal ? "$decimal\n" : "\n";
	}

	print "\n";
}
