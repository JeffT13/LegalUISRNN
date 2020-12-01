#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=integrate
#SBATCH --output=intchck_uisrnn.out

python LegalUISRNN/SCOTUS/testing/analyzeintegration.py