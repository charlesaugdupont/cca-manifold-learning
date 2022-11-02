#!/bin/bash

# Set job requirements
#SBATCH -t 04:00:00
#SBATCH -p normal

#SBATCH --nodes=1
#SBATCH --ntasks=16

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=c.a.dupont@uva.nl

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scikit-learn
pip install --user pandas

# Create output directory on scratch
mkdir "$TMPDIR"/output_dir_4 &

for i in `seq 48 63`; do
	# Execute python script
	python sim_framework.py "$TMPDIR"/output_dir_4 $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/output_dir_4 $HOME