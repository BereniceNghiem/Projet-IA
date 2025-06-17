#!/bin/bash
#SBATCH --job-name=cycleGAN_train            # Name of your job
#SBATCH --output=cycleGAN_train_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=cycleGAN_train_%j.err             # Error file
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
conda activate pytorch-cycleGAN

# Execute the Python script with specific arguments
srun python train.py \
  --epoch 0 \
  --n_epochs 100 \
  --batchSize 1 \
  --dataroot datasets/horse2zebra \
  --lr 0.0002 \
  --decay_epoch 50 \
  --height 384 \
  --width 512 \
  --input_nc 3 \
  --output_nc 3 \
  --cuda \
  --n_cpu 8 \

# Print job completion time
echo "Job finished at: $(date)"