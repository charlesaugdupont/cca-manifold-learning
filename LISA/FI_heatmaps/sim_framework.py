from itertools import product

import numpy as np
import pickle
import sys

from data_sim_module_lean import get_curve_samples, get_questionnaires, \
    						compute_mds, get_true_mds, \
							align_pca_mds, corr_between_coords

if __name__ == '__main__':

	output_dir = sys.argv[1]
	param_index = int(sys.argv[2])

	# parameter grid
	num_groups    = [5, 20, 50, 100]
	num_responses = [10, 25, 50, 100]
	num_questions = [4, 6]
	num_answers   = [3, 4]

	# extract parameter combination from input arg
	combinations = list(product(*[num_groups, num_responses, num_questions, num_answers]))
	K, num_responses, number_q, number_a = combinations[param_index]


	# range of sub-manifold parameters to test
	m_vals = np.linspace(1, 10, 10, dtype=int)

	# range of kappa values to test
	k_vals = np.linspace(1, number_a**number_q - 1, 10, dtype=int)

	# dimensions of FINE output
	dim = 2  

	correlation = np.zeros((len(m_vals), len(k_vals)))

	for m_idx, m in enumerate(m_vals):
		
		for k_idx, k in enumerate(k_vals):
			
			probs = get_curve_samples(number_q=number_q, number_a=number_a, samples=K, m=m, sin_angle=k-1)

			df = get_questionnaires(probs, count_answers=num_responses, number_q=number_q, number_a=number_a)

			# get theoretical embedding
			true_mds = get_true_mds(probs)

			# FINE
			_, mds_joint = compute_mds(df, dim=dim, compute_joint=True)

			# align the coordinates
			try:
				mds_joint = align_pca_mds(true_mds, mds_joint)
			except:
				continue

			# compute correlation between FI and theoretical embeddings
			fi_corr = corr_between_coords(true_mds, mds_joint)

			# store value
			correlation[m_idx][k_idx] = fi_corr

	# save results
	with open(output_dir + f"/{K}_{num_responses}_{number_q}_{number_a}.pickle", "wb") as f:
		pickle.dump(correlation, f)