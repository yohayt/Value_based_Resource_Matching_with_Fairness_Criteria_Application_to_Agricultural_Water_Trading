#!/bin/bash
#SBATCH --time=48:00:00

##SBATCH -n 1

# Load any modules you require:

echo "###################################"
echo "Command: $command"
echo "###################################"
eval "$command"
