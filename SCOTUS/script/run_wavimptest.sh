#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --job-name=wavdim
#SBATCH --output=wavdvec_test.out

python LegalUISRNN/SCOTUS/testing/wavdvec_importtest.py
echo completetest
