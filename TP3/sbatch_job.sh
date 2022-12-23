#!/bin/bash
#SBATCH --time=1:00
#SBATCH --partition=cpar
#SBATCH --constraint=k20

bin/k_means $1 $2
#nvprof bin/k_means $1 $2
