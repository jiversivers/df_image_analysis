#!/bin/bash

#SBATCH --job-name=mc_simulate
#SBATCH --partition=agpu06
#SBATCH --output=mc_main.txt
#SBATCH --error=mc_main.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdivers@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --qos=gpu


export OMP_NUM_THREADS=1

# load required module
module purge
module load python/anaconda-3.14

# Activate venv
conda activate /home/jdivers/.conda/envs/monte_carlo
echo $SLURM_JOB_ID

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job


echo "Copying files..."
mkdir scratch/$SLURM_JOB_ID/databases
rsync -avq $SLURM_SUBMIT_DIR/*.py /scratch/$SLURM_JOB_ID
rsync -avq /home/jdivers/data/hsdfm_database/hsdfm_data.db /scratch/$SLURM_JOB_ID/databases
rsync -avq /home/jdivers/df_image_analysis/my_modules /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

echo "Python script initiating..."
python3 monte_carlo_simulator.py

rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR/

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi
