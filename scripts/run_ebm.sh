#!/bin/bash
#$ -S /bin/bash
#$ -o /home/dsemchin/Progression_models_simulations/logs/output.log
#$ -e /home/dsemchin/Progression_models_simulations/logs/error.log
#$ -cwd
#$ -j y

CONFIG_FILE=$1
echo "Config file passed: $CONFIG_FILE" >> /home/dsemchin/Progression_models_simulations/logs/debug.log

PYTHON_EXEC=~/miniconda3/bin/python
SCRIPT_PATH=/home/dsemchin/Progression_models_simulations/scripts/run_mcmc_inference.py

echo "Running command: ${PYTHON_EXEC} ${SCRIPT_PATH} ${CONFIG_FILE}" >> /home/dsemchin/Progression_models_simulations/logs/debug.log
cmd="${PYTHON_EXEC} ${SCRIPT_PATH} ${CONFIG_FILE}"
eval $cmd