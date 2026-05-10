#!/bin/bash
#SBATCH --job-name=bond_conv
#SBATCH --account=commons
#SBATCH --partition=commons          # change to commons for longer runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-79         # 2 chi values x 40 g values
#SBATCH --output=/home/td62/EoS_project/rTEBD/bond_convergence/logs/bond_%A_%a.out
#SBATCH --error=/home/td62/EoS_project/rTEBD/bond_convergence/logs/bond_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
CHI_IDX=$((IDX / 40))
G_IDX=$((IDX % 40))

CHI_LIST=(64 128)
CHI=${CHI_LIST[$CHI_IDX]}
G=$(python3 -c "print(f'{1 + ${G_IDX} * 2.0 / 39:.6f}')")

PROJECT_DIR="$HOME/EoS_project/rTEBD"
RUN_DIR="$SHARED_SCRATCH/td62/bond_conv_chi${CHI}_gidx${G_IDX}_${SLURM_JOB_ID}"

mkdir -p "$PROJECT_DIR/bond_convergence/logs"
mkdir -p "$PROJECT_DIR/bond_convergence/results"
mkdir -p "$RUN_DIR"

# Copy just what the script needs
cp -r "$PROJECT_DIR/paraparticles" "$RUN_DIR/"
cp "$PROJECT_DIR/bond_convergence/bond_convergence.py" "$RUN_DIR/"
mkdir -p "$RUN_DIR/results"

module purge
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate codingDMRG

cd "$RUN_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

srun python bond_convergence.py --chi "$CHI" --g "$G"

# Copy result back to persistent location
cp "$RUN_DIR/results/"*.npz "$PROJECT_DIR/bond_convergence/results/"
echo "Done chi=$CHI g=$G, results in $PROJECT_DIR/bond_convergence/results/"