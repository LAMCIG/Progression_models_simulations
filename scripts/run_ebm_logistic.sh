#!/bin/bash
#PBS -S /bin/bash
#PBS -o /home/$dsemchin/log -j y
#PBS -l nodes=1:ppn=15  # Adjust the number of cores
#PBS -l walltime=12:00:00  # Adjust wall time based on the expected duration

cd /home/dsemchin/Progression_models_simulations/scripts

# Run the Python script
python3 run_ebm_logistic.py