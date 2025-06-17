#!/bin/bash
#SBATCH --job-name=comi_test            # Name of your job
#SBATCH --output=comi_test_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=comi_test_%j.err             # Error file
#SBATCH --partition=P100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=16G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-comi

# Execute the Python script with specific arguments
srun python test.py --cuda

# Print job completion time
echo "Job finished at: $(date)"