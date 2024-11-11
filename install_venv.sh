#!/bin/bash

# Name of the conda environment
env_name="AIFI_bootcamp"

# Name of your requirements.txt file
requirements_file="requirements.txt"

# Create the conda environment
conda create -n $env_name python=3.10

# Activate the conda environment
conda activate $env_name

# Install the requirements
pip install -r $requirements_file

# Print a success message
echo "Setup is complete."
