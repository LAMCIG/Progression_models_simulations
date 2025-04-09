#!/bin/bash
#PBS -S /bin/bash
#PBS -o /ifshome/$USER/log -j y
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=02:00:00
#PBS -t 0-23

cd $PBS_O_WORKDIR

# parameter grid
noise_levels=(0.0 0.1 0.2 0.3)
n_biomarkers_list=(5 10 20 40)
jacobian_flags=(True False)

# compute index from PBS_ARRAYID (0-23)
idx=${PBS_ARRAYID}

# decode grid parameters
noise_idx=$((idx / 6))       # 0-3
bio_idx=$(((idx % 6) / 2))   # 0-3
jac_idx=$((idx % 2))         # 0-1

noise=${noise_levels[$noise_idx]}
n_biomarkers=${n_biomarkers_list[$bio_idx]}
jacobian=${jacobian_flags[$jac_idx]}

# run experiment
python3 scripts/run_experiment.py --noise_level $noise --n_biomarkers $n_biomarkers --use_jacobian $jacobian