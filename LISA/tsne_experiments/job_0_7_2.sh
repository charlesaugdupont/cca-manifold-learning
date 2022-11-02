#!/bin/bash

# Set job requirements
#SBATCH -t 08:00:00
#SBATCH -p normal

#SBATCH --nodes=1
#SBATCH --ntasks=8

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=c.a.dupont@uva.nl

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scikit-learn
pip install --user pandas
pip install --upgrade --user threadpoolctl

# Create output directory on scratch
mkdir "$TMPDIR"/tsne_071 &

for i in `seq 0 7`; do
	# Execute python script
	python tsne_experiment.py "$TMPDIR"/tsne_071 $i 1 &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/tsne_071 $HOME