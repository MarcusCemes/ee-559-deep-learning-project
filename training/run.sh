#!/bin/bash
#SBATCH --job-name=finetune-hateBERT
#SBATCH --partition=gpu
#SBATCH --time=10:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --chdir=/scratch/izar/cemes
#SBATCH --output=/scratch/izar/cemes/%x-%j.out

source .venv/bin/activate

srun python3 -m training.app
