#!/bin/bash
#$ -S /bin/bash         # rpecifies the shell for the job
#$ -o /path/to/log      # redirect output to a log directory
#$ -e /path/to/log      # redirect error messages to a log directory
#$ -cwd                 # run the job from the current directory
#$ -j y                 # merge standard error and output into a single file

# JSON file
CONFIG_FILE=$1
PYTHON_EXEC=/home/dsemchin/miniconda3/bin/python
SCRIPT_PATH=/home/dsemchin/Progression_models_simulations/scripts/run_mcmc_inference.py
cmd="${PYTHON_EXEC} ${SCRIPT_PATH} ${CONFIG_FILE}"
echo "Running command: $cmd"
eval $cmd