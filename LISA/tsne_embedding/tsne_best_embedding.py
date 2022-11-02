from data_sim_module_lean import get_curve_samples, get_questionnaires, \
                            	 get_true_mds, align_pca_mds, corr_between_coords, \
								 multi_partite_distance

import pandas as pd
import numpy as np
import pickle
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE


def tsne_alg(df, columns, dim=2, perplexity=1, learning_rate=50, **kwargs):
    """ Computes a T-SNE on the samples, to test how this performs in
    comparison with FINE.
    """
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1

    data = OneHotEncoder(sparse=False).fit_transform(df[columns])

    tsne = TSNE(n_components=dim, perplexity=perplexity, 
				learning_rate=learning_rate, init="pca", **kwargs).fit_transform(data)
    subaks = sorted(df.name_1.unique())
    tsne_coords = np.zeros(shape=(len(subaks), dim))
    for i, s in enumerate(subaks):
        # Compute the position of the Subak in the PCA space
        ind = df[df.name_1 == s].index
        tsne_coords[i, :] = np.mean(tsne[ind], axis=0)

    return tsne_coords


if __name__ == '__main__':

	output_dir = sys.argv[1]
	param_index = int(sys.argv[2])

	m = 3
	dim = 2
	number_q = 8
	number_a = 3
	N = number_a**number_q

	# theoretical embeddings
	kappa     = [1, 2, N-1, 1, 2, N-1]
	samples   = [20, 20, 20, 50, 50, 50]
	responses = [25, 25, 25, 50, 50, 50]
	k, num_samples, num_responses = kappa[param_index], samples[param_index], responses[param_index]

	# set perplexity and learning rate
	p  = 1
	lr = 50

	best_corr   = -10
	best_coords = None
	best_KLs    = None

	for iteration in range(30):
		probs = get_curve_samples(number_q=number_q, number_a=number_a, 
								  samples=num_samples, m=m, sin_angle=k-1)
		KLs = multi_partite_distance(probs, Nq=number_q, Na=number_a)
		df = get_questionnaires(probs, count_answers=num_responses, number_q=number_q, number_a=number_a)
		true_mds = get_true_mds(probs)
		df.name_1 = df.name_1.astype(int)
		tsne = tsne_alg(df, df.columns.drop("name_1"), dim=dim, perplexity=p, learning_rate=lr)
		tsne = align_pca_mds(true_mds, tsne)
		tsne_corr = corr_between_coords(true_mds, tsne)
		if tsne_corr > best_corr:
			print(tsne_corr)
			best_corr = tsne_corr
			best_coords = (tsne, true_mds)
			best_KLs = KLs
			if best_corr >= 0.93:
				break

	# save results
	with open(output_dir + f"/{param_index}.pickle", "wb") as f:
		pickle.dump({"corr":best_corr, "true_coords":best_coords[1], 
					 "tsne_coords":best_coords[0], "KLs":best_KLs}, f)