#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --job-name=tunecase50
#SBATCH --output=casewavtune50.out

python LegalUISRNN/SCOTUS/SVE/sco50/tune_sco50wav_case.py