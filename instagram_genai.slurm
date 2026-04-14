#!/bin/bash
#SBATCH -J instagram_genai    # Job name
#SBATCH -o run_%j.out    # Standard output
#SBATCH -e run_%j.err    # Error output
#SBATCH -p gpu-a100-small           # GPU partition
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (1 CPU)
#SBATCH -t 02:00:00             # Time limit (2 hour)
#SBATCH -A CDA24004         # Your actual allocation code

#SBATCH --mail-user=anthonyang2027@utexas.edu
#SBATCH --mail-type=BEGIN,END,FAIL,CANCEL

cd $WORK/IG_GenAI_Project
module load python3
module load cuda
source myenv/bin/activate
python Image_Processing_Report.py
