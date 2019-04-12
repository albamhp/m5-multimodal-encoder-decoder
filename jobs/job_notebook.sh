#!/usr/bin/env bash
#SBATCH --job-name notebook
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/m5-multimodal-encoder-decoder
#SBATCH --output ../logs/%x_%u_%j.out

source /home/grupo06/venv/bin/activate
python jupyter-notebook --no-browser --port=7810 notebook${SLURM_JOB_ID}