import numpy as np
from genotype_caller import call_genotype, ALLELES


def somatic_vaf(n_obs, error_rate, gt=None):
    """
    Compute the somatic variant allele frequency at a given position
    from the allele counts and expected error rate at that position.

    :param n_obs: array (len = 4)
    An array of length 4 representing
    the allele count for alleles A, T, G, C in that order.
    :param error_rate: float
    A number between 0 and 1 representing the expected error rate
    at the position where the allele count was observed.
    :return: array
    Return an array of 4 floats representing
    the somatic variant allele frequencies of A, T, G, C in that order.
    """
    if not gt:
        gt = call_genotype(n_obs)[0]  # germline genotype
    vaf = np.zeros(4)
    if gt[0] == gt[1]:  # homozyguous case
        germline_allele_index = ALLELES.index(gt[0])
        variant_allele_indices = [i for i in range(4) if i != germline_allele_index]
        n_germline = n_obs[germline_allele_index]
        if n_germline <= 0:
            return vaf
        variant_allele_ratio = n_obs[variant_allele_indices] / n_germline
        det = 1 + np.sum(variant_allele_ratio)
        vaf[variant_allele_indices] = 1 / det * (1 - 3 / 2 * error_rate) * variant_allele_ratio - error_rate / 4
    else:  # heterozyguous case. A bit of boilerplate here, but it is good for performance consideration.
        germline_allele_indices = [ALLELES.index(gt[0]), ALLELES.index(gt[1])]
        variant_allele_indices = [i for i in range(4) if i not in germline_allele_indices]
        n_germline = np.sum(n_obs[germline_allele_indices])
        if n_germline <= 0:
            return vaf
        variant_allele_ratio = n_obs[variant_allele_indices] / n_germline
        det = 1 + np.sum(variant_allele_ratio)
        vaf[variant_allele_indices] = 1 / det * variant_allele_ratio - error_rate / 4
    return vaf

