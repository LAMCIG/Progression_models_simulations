#!/bin/bash
#PBS -N EBM_job                    
#PBS -l nodes=1:ppn=10              
#PBS -l mem=16gb
#PBS -o /home/dsemchin/EBM_output.log  # output file
#PBS -e /home/dsemchin/EBM_error.log   # error file
#PBS -j oe                         
#PBS -V                           

cd /home/dsemchin/Progression_models_simulations/scripts
bash run_ebm.sh /home/dsemchin/Progression_models_simulations/scripts/configs/noise_acp.json
