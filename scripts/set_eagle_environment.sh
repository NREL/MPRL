module purge
MODULES=modules
COMPILER=gcc-7.4.0
module unuse ${MODULEPATH}
module use /nopt/nrel/ecom/hpacf/binaries/${MODULES}
module use /nopt/nrel/ecom/hpacf/compilers/${MODULES}
module use /nopt/nrel/ecom/hpacf/utilities/${MODULES}
module use /nopt/nrel/ecom/hpacf/software/${MODULES}/${COMPILER}

module load gcc
module load git
module load mpich
module load texlive/live

MINICONDA_DIR=$HOME/miniconda3
if [ -d "$MINICONDA_DIR" ]; then
    . ${MINICONDA_DIR}/etc/profile.d/conda.sh
fi
conda activate /projects/exact/mhenryde/.conda/MPRL

export ranks_per_node=36
export OMP_NUM_THREADS=1  # Max hardware threads = 4

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"
