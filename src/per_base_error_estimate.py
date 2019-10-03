import numpy as np
from genotype_caller import call_genotype, ALLELES


def error_estimate(n_obs, gt=None):
    if not gt:
        gt = call_genotype(n_obs)[0]
    n = np.sum(n_obs)
    if gt[0] == gt[1]:
        allele_index = ALLELES.index(gt[0])
        p = n_obs[allele_index] / n
        return 4 / 3 * (1 - p)
    else:
        p = (n_obs[ALLELES.index(gt[0])] + n_obs[ALLELES.index(gt[1])]) / n
        return 2 * (1 - p)
