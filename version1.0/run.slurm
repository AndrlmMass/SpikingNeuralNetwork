#!/bin/bash -x
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolaih@simula.no
#SBATCH --job-name="sscp-2024"
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

source ./dev_env/bin/activate

srun python main/Main_SNN.py