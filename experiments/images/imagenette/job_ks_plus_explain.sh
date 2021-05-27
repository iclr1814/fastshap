#!/bin/bash
#SBATCH --partition=gpu4_short
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --job-name=ks_plus_explain
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1000
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1


python ks_plus_explain.py
