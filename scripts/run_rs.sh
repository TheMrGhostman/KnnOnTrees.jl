#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpufast
#SBATCH --mem=5G


IDX=$1
RS=$2
SEED=$3

module load Julia/1.9.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

julia --project ./knn_rs_step.jl ${IDX} ${RS} ${SEED}
