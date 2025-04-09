#!/bin/bash
#PBS -S /bin/bash
#PBS -o /ifshome/$USER/log
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=02:00:00
#PBS -t 0-23

# move to working directory
cd $PBS_O_WORKDIR

# define grid
noise_levels=(0.0 0.1 0.2 0.3)
n_biomarkers_list=(5 10 20 40)
jacobian_flags=(True False)

# decode PBS_ARRAYID
idx=${PBS_ARRAYID}
noise_idx=$((idx / 6))         # 0 to 3
bio_idx=$(((idx % 6) / 2))     # 0 to 3
jac_idx=$((idx % 2))           # 0 to 1

# extract values
noise=${noise_levels[$noise_idx]}
n_biomarkers=${n_biomarkers_list[$bio_idx]}
jacobian=${jacobian_flags[$jac_idx]}

# run experiment
python3 scripts/run_em_experiment.py --noise_level $noise --n_biomarkers $n_biomarkers --use_jacobian $jacobian
