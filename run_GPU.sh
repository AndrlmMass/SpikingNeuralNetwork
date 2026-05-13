#!/bin/bash
#SBATCH --job-name=snn_job
#SBATCH --output=results/out_%j.log
#SBATCH --error=results/err_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

cd ~/projects/src

if [ ! -f noise_env.sif ]; then
    singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
    singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
fi

singularity exec --nv noise_env.sif conda run -n noise_env python main.py