#!/bin/bash
#SBATCH --time=1:00
#SBATCH --partition=cpar
#SBATCH --constraint=k20

nvprof --cpu-profiling on --cpu-profiling-mode top-down bin/k_means $1 $2
