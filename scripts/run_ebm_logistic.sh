#!/bin/bash
#PBS -S /bin/bash
#PBS -o /home/$dsemchin/log -j y
#PBS -j oe 
#PBS -l nodes=1:ppn=15
#PBS -l walltime=12:00:00 

cd /home/dsemchin/Progression_models_simulations/scripts

# Run the Python script
python3 run_ebm_logistic.py