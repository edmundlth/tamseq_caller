import numpy as np
import pandas as pd
from scipy.stats import entropy


GENOTYPES_ALLELE_PROB = {
    "AA": np.array([1, 0, 0, 0]),
    "TT": np.array([0, 1, 0, 0]),
    "GG": np.array([0, 0, 1, 0]),
    "CC": np.array([0, 0, 0, 1]),
    "AT": np.array([.5, .5, 0, 0]),
    "AG": np.array([.5, 0, .5, 0]),
    "AC": np.array([.5, 0, 0, .5]),
    "TG": np.array([0, .5, .5, 0]),
    "TC": np.array([0, .5, 0, .5]),
    "GC": np.array([0, 0, .5, .5])
}

ALLELES = ['A', 'T', 'G', 'C']


def call_genotype(n_obs):
    """

    :param n_obs: array (len = 4)
    An array of length 4 representing
    the allele count for alleles A, T, G, C in that order.

    :return: (String, float, float)
    Return the genotype, the minimum and the second smallest
     Kullback-Leiber divergence among the 10 competing genotype hypothesis.
    """
    n_obs = np.array(n_obs)
    n = np.sum(n_obs)  # Sum of allele counts
    k = n_obs.shape[0]  # Number of allele type should equal 4
    p_obs = (n_obs + 1) / (n + k)  # inline laplace_smoothing(n_obs)

    best_gt = None
    smallest_div_kl = np.inf
    second_smallest_div_kl = np.inf
    for gt, probs in GENOTYPES_ALLELE_PROB.items():
        n_exp = n * probs
        p_exp = (n_exp + 1) / (n + k)   # inline laplace_smoothing(n_exp)
        divergence = entropy(p_obs, p_exp)
        if divergence < smallest_div_kl:
            best_gt = gt
            second_smallest_div_kl = smallest_div_kl
            smallest_div_kl = divergence
        elif divergence < second_smallest_div_kl:
            second_smallest_div_kl = divergence
    return best_gt, smallest_div_kl, second_smallest_div_kl


def laplace_smoothing(n_obs):
    return (n_obs + 1) / (np.sum(n_obs) + len(n_obs))


def g_statistics(n_obs, n_exp):
    return 2 * np.add.reduce(n_obs * np.log(n_obs / n_exp))
