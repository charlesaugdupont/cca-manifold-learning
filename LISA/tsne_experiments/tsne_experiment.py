from data_sim_module_lean import get_curve_samples, get_questionnaires, \
                            	 get_true_mds, align_pca_mds, corr_between_coords

from itertools import product
import pandas as pd
import numpy as np
import pickle
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE


def tsne_alg(df, columns, dim=2, perplexity=3, learning_rate=20, **kwargs):
    """ Computes a T-SNE on the samples, to test how this performs in
    comparison with FINE.
    """
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1

    # data = StandardScaler().fit_transform(df[columns])
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
	k_index = int(sys.argv[3])

	dim = 2
	m = 3
	number_q = 8
	number_a = 3
	N = number_a**number_q

	kappas = [2, N-1]
	num_samples   = 20
	num_responses = 25

	p_vals  = [1, 2, 3, 4]
	lr_vals = [10, 20, 50, 100]
	
	# extract parameter combination from input arg
	combinations = list(product(*[p_vals, lr_vals]))
	p, lr = combinations[param_index]
	k = kappas[k_index]

	results = []
	for iteration in range(30):
		print(iteration)
		probs = get_curve_samples(number_q=number_q, number_a=number_a, 
								  samples=num_samples, m=m, sin_angle=k-1)
		df = get_questionnaires(probs, count_answers=num_responses, number_q=number_q, number_a=number_a)
		true_mds = get_true_mds(probs)
		df.name_1 = df.name_1.astype(int)
		tsne = tsne_alg(df, df.columns.drop("name_1"), dim=dim, perplexity=p, learning_rate=lr)
		tsne = align_pca_mds(true_mds, tsne)
		tsne_corr = corr_between_coords(true_mds, tsne)
		results.append(tsne_corr)

	# save results
	with open(output_dir + f"/{p}_{lr}_{k}.pickle", "wb") as f:
		pickle.dump(results, f)
