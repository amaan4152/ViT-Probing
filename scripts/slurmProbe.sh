#!/bin/bash
#
#SBATCH --job-name=ViT_PROBING
#SBATCH --output=/zooper2/amaan.rahman/output.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=3gb

source .bashrc
cd ViT-Probing/python/
make automate_train_probes