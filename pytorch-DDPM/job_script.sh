#!/bin/bash
#SBATCH --job-name=ddpm_train              # Nom du job
#SBATCH --output=ddpm_train_%j.out         # Fichier de sortie
#SBATCH --error=ddpm_train_%j.err          # Fichier d'erreur
#SBATCH --partition=P100                   # Partition GPU (adaptée à ton infra)
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --cpus-per-task=8                  # 8 CPUs
#SBATCH --mem=32G                          # RAM demandée
#SBATCH --time=24:00:00                    # Durée max du job

# === Infos système ===
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# === Activation de l'environnement ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-ddpm  # Remplace par l'env contenant `denoising_diffusion_pytorch`

# === Lancement du script ===
srun python train.py

# === Fin du job ===
echo "Job finished at: $(date)"