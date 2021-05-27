#!/bin/bash
#SBATCH --partition=gpu4_short
#SBATCH --time=00:5:00
#SBATCH --mem=16G
#SBATCH --job-name=ks_explain
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1000
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1


source /gpfs/home/nj594/.bashrc
source /gpfs/home/nj594/.bash_profile
conda activate tf23

python ks_explain.py
