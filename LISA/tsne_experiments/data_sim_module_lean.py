#!/usr/bin/env python3
# encoding: utf-8

"""
Data simulation module
----------------------

This module combines the scripts simulate_data.py, hypersphere.py, and spherical.py
to create a single module of all data simulation related algorithms.

 - Netherlands eScience Center
"""

# Standard imports for simulate_data
import numpy as np
import pandas as pd

# Imports for hypersphere.py
from collections import defaultdict
from itertools import combinations

from numpy import cos, linspace, ones, pi, sin, zeros
from scipy.spatial.distance import pdist
from sklearn.manifold import MDS

"""
Simulate data for sparse FINE
-----------------------------

This script simulates data for Subak analysis which should enable me to verify
different statistical properties of the results, test how sparsity affects
FINE, etc...

Usage
------
simulate_date(N_params, N_questions, N_answers, N_groups, N_answers_per_group)

# N_params - Number of parameters in the model
# N_questions - Number of questions in the questionnaire
# N_answers - Number of answers per question
# N_groups - Number of groups (Subaks)
# N_answers_per_group -  Number of of answers per group (fixed)

Written by
----------

Omri Har-Shemesh,
University of Amsterdam

"""

"""
Hyperspehere.py algorithms
"""

def mds_wrapper(dist_mat, dim=2, return_stress=False):
    """ Runs the MDS algorithm and returns the coordinates. The reason for
    this wrapper is so that the parameters given to MDS (n_jobs, n_init, etc..)
    will be consistent across all runs of the algorithm.

    Args:
        dist_mat (ndarray): Distance matrix to use in the MDS computation.

    Kwargs:
        dim (int, 2): The dimensionality of the MDS.

        return_stress (bol, False): Should the stress of the embedding also be
            returned?

    Returns: The coordinates calculated using the MDS algorithm.

    """
    mds = MDS(n_components=dim, dissimilarity='precomputed', n_init=15,
            max_iter=1000, n_jobs=7).fit(dist_mat)

    if return_stress:
        return mds.embedding_, mds.stress_
    return mds.embedding_

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Taken from:
    http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy

    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. They must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. Setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform


def compute_mds(df, dim=2, compute_joint=False, columns=None, return_stress=False):
    """ Computes the MDS embedding of the questionnaire results based on
    Fisher information distance.

    Args:
        df (DataFrame): DF containing the questionnaire responses.

    Kwargs:
        dim (int, 2): The dimensionality of the returned MDS.

        compute_join (Bool, True): Should the MDS compute also the distances
            from the joint distribution (assuming independence)?

        columns (list-like, None): If not None, a list of columns to take into
            account in the computation.

        return_stress (Bool, False): If true, returns the calculated stress for
            the MDS as well

    Returns: ndarray of coordinates. If compute_join, returns both coordinates
             for mean and for joint MDS.

    """
    # Estimate probabilities from questionnaire
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(df.name_1, how="right")
    if columns is None:
        columns = df.columns.drop("name_1")
    for c in columns:
        ind_df = ind_df.join(pd.get_dummies(df[c], prefix=c))

    # Turn this into a probabilities matrix for each possible response
    response_count = df.groupby(df.name_1).apply(lambda x: len(x))
    sum_df = ind_df.groupby(ind_df.name_1).sum()
    prob = sum_df.divide(response_count, "index")

    # Compute the square root of the probabilities
    sqrt_prob = np.sqrt(prob)

    # Compute the joint PDF assuming independence
    question_dict = defaultdict(list)
    for c in sqrt_prob.columns:
        question_dict["_".join(c.split("_")[:-1])].append(c)

    # prods = list(product(*question_dict.values()))
    # for i in range(len(prods)):
        # joint_mult = pd.DataFrame(sqrt_prob[list(prods[i])].prod(axis=1),
                # columns=[i])
        # if i == 0:
            # joint = joint_mult
        # else:
            # joint.ix[:,i] = joint_mult

    # Get all pairs of Subaks (without return)
    all_pairs = combinations(sqrt_prob.index, 2)

    res = {}
    res_joint = {}
    for c1, c2 in all_pairs:
        s1 = sqrt_prob.loc[c1]
        s2 = sqrt_prob.loc[c2]

        # To compute the Fisher distance, we use
        # FI = arccos(sum(sqrt(p_i*q_i))) for each question separately.
        pq = s1 * s2

        # This gives the sum over the multiplication performed on a per
        # question basis.
        questions = pq.groupby(lambda x: "_".join(x.split("_")[:-1])).sum()
        # Fix when rounding errors cause it to be slightly larger than 1.0
        questions[questions > 1] = 1.0
        FIs = np.arccos(questions)
        FI = FIs.mean()

        # joints = (joint.ix[c1] * joint.ix[c2]).sum()
        # if joints > 1.0: joints = 1.0
        # FI_joint = np.arccos(joints)

        # Fast joint
        FI_joint = np.arccos(questions.prod())

        c1i = int(c1)
        c2i = int(c2)

        if c1i in res:
            res[c1i][c2i] = FI
            res_joint[c1i][c2i] = FI_joint
        else:
            res[c1i] = {c2i: FI, c1i: 0.0}
            res_joint[c1i] = {c2i: FI_joint, c1i: 0.0}
        if c2i in res:
            res[c2i][c1i] = FI
            res_joint[c2i][c1i] = FI_joint
        else:
            res[c2i] = {c1i: FI, c2i: 0.0}
            res_joint[c2i] = {c1i: FI_joint, c2i: 0.0}

    dist_mat = pd.DataFrame(res).sort_index().sort_index(axis=1)
    dist_mat_joint = pd.DataFrame(res_joint).sort_index().sort_index(axis=1)

    # Perform the MDS
    if return_stress:
        mds_coords, mds_stress = mds_wrapper(dist_mat, dim=dim,
                return_stress=True)
        mds_coords_joint, joint_stress = mds_wrapper(dist_mat_joint, dim=dim,
                return_stress=True)
    else:
        mds_coords = mds_wrapper(dist_mat, dim=dim)
        mds_coords_joint = mds_wrapper(dist_mat_joint, dim=dim)

    if compute_joint:
        if return_stress:
            return mds_coords, mds_stress, mds_coords_joint, joint_stress
        else:
            return mds_coords, mds_coords_joint
    if return_stress:
        return mds_coords, mds_stress
    return mds_coords


def corr_between_coords(coords1, coords2):
    """ Computes the correlation between two sets of coordinates via the
    distance matrix

    Args:
        coords1 (ndarray): First array of coordinates
        coords2 (ndarray): Second array of coordinates

    Returns: Correlation between the two coordinates

    """
    dm1 = pdist(coords1)
    dm2 = pdist(coords2)

    return np.corrcoef(dm1, dm2)[0, 1]

def draw_questions(sqrt_probs, number_q=3, number_a=3, count=25):
    """ Draw answers from the join probability defined on the sphere.
        Currently assumes number of questions and answers is 3

    Args:
        number_q (TODO): TODO
        number_a (TODO): TODO
        sqrt_probs (TODO): TODO

    Kwargs:
        count: Number of answers to draw

    Returns: A list of responses based on the sqrt_probs provided

    """
    assert len(sqrt_probs) == number_a ** number_q

    rands = np.random.rand(count)
    res = []
    probs = sqrt_probs ** 2
    probs = probs.cumsum()
    ans_dict = get_ans_dict(number_q, number_a)
    for i in range(count):
        if rands[i] == 1.0:
            num = number_a ** number_q - 1
        else:
            num = np.argwhere(probs - rands[i] > 0)[0][0]

        # Convert number to answers
        res.append(list(ans_dict[num]))

    return res

def get_coords(angles):
    """ Computes the coordinates of a point on a unit hypersphere whose spherical
    coordinates (angles) are given.

    Args:
        angles (array): An array of angles

    Returns: Coordinates of the point thus specified

    """
    dim = angles.shape[0] + 1
    count = angles.shape[1]
    coords = ones(shape=(dim, count))

    for i in range(dim - 1):
        coords[i] *= cos(angles[i])
        coords[i+1:dim] *= sin(angles[i])

    return coords

def get_ans_dict(number_q=3, number_a=3):
    """ Computes a dictionary that can convert a number between 0 and
        number_a^number_q - 1 and responses to a questionnaire.

    Kwargs:
        number_q (TODO): TODO
        number_a (TODO): TODO

    Returns: TODO

    """
    max_num = number_a ** number_q
    res = {}
    for i in range(max_num):
        ans = []
        num = i
        for j in range(number_q - 1):
            exp = number_a ** (number_q - 1 - j)
            ans.append(num // exp)
            num = num % exp
        ans.append(num)
        res[i] = ans
    return res

def get_questionnaires(probs, number_q=3, number_a=3, count_answers=25):
    """ Produces a dataframe with simulated questionnaires, one for each
    probability distribution ("Subak")

    Args:
        probs (TODO): TODO

    Kwargs:
        count_answers (TODO): TODO

    Returns: TODO

    """
    q = []
    for i in range(probs.shape[0]):
        ans = draw_questions(probs[i], count=count_answers, number_q=number_q,
                number_a=number_a)
        for a in ans:
            a.insert(0, i + 1)
            q.append(a)

    columns = ['name_1']
    for i in range(number_q):
        columns.append("%s_%d" % (chr(ord("A")+i), i+2))

    q = pd.DataFrame(q, columns=columns)
    q.name_1 = q.name_1.astype(str)
    return q


def get_curve_samples(number_q=3, number_a=3, count=1000, samples=20, m=2,
        inds=None, random=False, rep=1, return_t=False, sin_angle=0,
        sin_angle_2=None):
    """ Returns a list of PDFs as samples from a curve.

    Kwrgs:
        number_q (int, 3): Number of questions to simulate in the
            questionnaire.

        number_a (int, 3): Number of possible answers per question.

        count (int, 1000): Number of points along the curve to calculate.

        samples (int, 20): Number of points to draw uniformly from the count
            number of points.

        inds (list-like, None): If not None, a list of indices to take from
            the count number of samples computed, instead of uniformly using
            samples.

        random (Bool, False): If set to true, sample the samples randomly form
            the curve, rather than take them at set intervals.

        rep (int, 1): How many sets of samples to return, in case sampling
            randomly rather than at set intervals.

        return_t (Bool, False): If true, the function will also return the
            value of t for each of the samples. This is necessary if one
            wants to compute the distance along the curve (using the analytic
            expression for the Fisher information).

        sin_angle (int, 0): Which angle to set to be the sine squared of t.
            Each selection gives a different type of curve.


    Returns: A two dimensional array of coordinates

    """
    dim = number_a ** number_q
    angles = zeros(shape=(dim - 1, count))

    for i in range(0, dim-1):
        angles[i, :] = linspace(0, pi/2, count)

    x = linspace(0, 1, count)
    angles[sin_angle, :] = (pi/2) * sin(x * pi * m)**2

    coords = get_coords(angles)

    if inds != None:
        if return_t:
            ts = x[inds]
            return coords[:, inds].T, ts
        return coords[:, inds].T

    if random:
        if rep == 1:
            inds = np.random.choice(list(range(count)), size=samples, replace=False)
            inds = sorted(inds)
            if return_t:
                ts = x[inds]
                return coords[:, inds].T, ts
            return coords[:, inds].T
        coords_arr = []
        for i in range(rep):
            inds = np.random.choice(list(range(count)), size=samples, replace=False)
            inds = sorted(inds)
            if return_t:
                ts = x[inds]
                coords_arr.append([coords[:, inds].T, ts])
            coords_arr.append(coords[:, inds].T)
        return coords_arr

    if return_t:
        ts = x[::count//samples]
        return coords[:,::count//samples].T, ts
    return coords[:,::count//samples].T


def get_true_mds(probs, dim=2):
    """ Returns the mds embedding of the "true" distance matrix based on the
    complete joint pdf.

    Args:
        probs (TODO): TODO

    Returns: TODO

    """
    true_dist_mat = get_true_dist_mat(probs)
    true_mds = mds_wrapper(true_dist_mat, dim=dim)

    return true_mds


def get_true_dist_mat(probs):
    """ Computes the "ground truth" distance matrix from the probabilities
    themselves

    Args:
        probs (TODO): TODO

    Returns: TODO

    """
    samples = probs.shape[0]
    true_dist_mat = zeros(shape=(samples, samples))
    for i, j in combinations(range(samples), 2):
        ij_mult = (probs[i] * probs[j]).sum()
        true_dist_mat[i, j] = np.arccos(ij_mult)
        true_dist_mat[j, i] = true_dist_mat[i, j]

    return true_dist_mat


def align_pca_mds(pca, mds):
    """ Aligns the coordinates obtained by PCA and those obtained by MDS
        so that they appear the closest. This is done by solving the
        Procrostean problem.

    Args:
        pca (ndarray): Array of coordinates obtained from PCA
        mds (ndarray): Array of coordinates obtained by MDS.

    Returns: The MDS after scaling, rotating and possibly reflecting so that
        it will best fit the PCA result. This is done by using the solution
        to the Procrustean problem.

    """

    d, mds_new, tform = procrustes(pca, mds)
    return mds_new