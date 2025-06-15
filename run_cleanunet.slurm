#!/bin/bash

#SBATCH --job-name=cleanunet_voicebank
#SBATCH --output=logs/cleanunet_%j.out
#SBATCH --error=logs/cleanunet_%j.err
#SBATCH --time=24:00:00                    # 24 hours max runtime
# #SBATCH --partition=work                 # Using default partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                  # 8 CPU cores
#SBATCH --mem=32G                          # 32GB RAM
#SBATCH --gres=gpu:1                       # 1 GPU (change to gpu:2 for multi-GPU)

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Create logs directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load necessary modules (adjust to your cluster)
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate denoise_env

# Verify GPU is available
echo "GPU Info:"
nvidia-smi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to project directory
cd /home/emilybederov/CleanUNet

# Single GPU training
echo "Starting CleanUNet training..."
python train.py -c configs/VOICEBANK-large-full.json

# For multi-GPU training, use this instead:
# python distributed.py -c configs/VOICEBANK-large-full.json

echo "Training completed at: $(date)"

# Optional: Copy checkpoints to permanent storage
# cp -r ./exp/VOICEBANK-large-full/checkpoint /scratch/your_username/cleanunet_checkpoints_$(date +%Y%m%d_%H%M%S)

echo "Job finished successfully!"
