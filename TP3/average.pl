#!/usr/bin/env perl

use strict;
use warnings;

foreach my $fn (@ARGV){

	my $fh = undef;

	unless(-f $fn and open $fh, '<', $fn){
		warn "$!";
		next;
	}


	my @KEYS = ('cycles', 'instructions', 'seconds time');

	my %sums   = ( $KEYS[0] => 0, $KEYS[1] => 0, $KEYS[2] => 0 );
	my %counts = ( $KEYS[0] => 0, $KEYS[1] => 0, $KEYS[2] => 0 );


	while(<$fh>){

		if(m/\s*(.+?)\s+($KEYS[0]|$KEYS[1]|$KEYS[2])/){
	
			my $one = $1;
			my $two = $2;

			$one =~ s/,//g;

			$sums{$two} += $one;
			$counts{$two}++;
		}
	}

	for my $key (@KEYS){
		$sums{$key} /= $counts{$key};
	}

	print "$fn:\n";

	
	#foreach my $key (@KEYS){

	my $key = 'seconds time';
	my $rounded = $sums{$key};


	$rounded =~ s/(\d+)(\.\d+)?/$1/;
	my $decimal = $2;
	
	while($rounded =~ s/(\d+)(\d\d\d)/$1\,$2/){}
	
	print "\t$key => $rounded", $decimal ? "$decimal" : '', "\n";
	#}

	print "\n";
}
