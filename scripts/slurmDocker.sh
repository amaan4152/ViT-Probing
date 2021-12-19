#!/bin/bash
#
#SBATCH --job-name=TRAIN_PROBES
#SBATCH --output=/zooper2/amaan.rahman/output.txt
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=5gb

source .bashrc
cd ViT-Probing/
make automate_train_probes
