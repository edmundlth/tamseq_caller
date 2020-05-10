import numpy as np
from genotype_caller import call_genotype, ALLELES
from scipy import stats
from scipy.stats import chi2
from scipy.special import gamma, polygamma, digamma
from scipy.optimize import root_scalar

MASKS = np.unpackbits(np.arange(1, 2 ** 4, dtype=np.uint8).reshape(-1, 1), axis=1)[:, 4:].astype(bool)


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
    else:  # heterozygous case. A bit of boilerplate here, but it is good for performance consideration.
        germline_allele_indices = [ALLELES.index(gt[0]), ALLELES.index(gt[1])]
        variant_allele_indices = [i for i in range(4) if i not in germline_allele_indices]
        n_germline = np.sum(n_obs[germline_allele_indices])
        if n_germline <= 0:
            return vaf
        variant_allele_ratio = n_obs[variant_allele_indices] / n_germline
        det = 1 + np.sum(variant_allele_ratio)
        vaf[variant_allele_indices] = 1 / det * variant_allele_ratio - error_rate / 4
    return vaf


def variant_caller_broadcast(df_data):
    """
    :param df_data: pd.DataFrame
    With columns: A, T, G, C, error_estimate
    :return: pd.DataFrame
    A copy of df_data with VAF prediction appended as new columns
    """
    df = df_data.loc[df_data[ALLELES].sum(axis=1) > 0]
    n_obs = df[ALLELES]

    n_obs_sqrt = np.sqrt(n_obs)
    vaf = n_obs_sqrt.div(np.sum(n_obs_sqrt, axis=1), axis='index')

    non_zero_error_rate = df["error_estimate"] > 0
    error_rate = df[non_zero_error_rate]["error_estimate"]
    k = np.count_nonzero(n_obs[non_zero_error_rate] == 0, axis=1)
    a = vaf[non_zero_error_rate].mul((4 * (2 / error_rate - 1) - k), axis='index')
    b = 1 / 8 * (np.sqrt(1 + a ** 2) - 1)
    vaf[non_zero_error_rate] = b.mul(error_rate, axis='index')
    return df.join(vaf, how='left', rsuffix='_VAF')


def variant_caller(n_obs, error_rate):
    if np.count_nonzero(n_obs) == 0:
        return np.zeros(4), 0
    elif error_rate == 0:
        n = np.sum(n_obs)
        return n_obs / n, n
    n = np.sum(n_obs)
    e = error_rate / 4
    vaf = n_obs / n - e

    if np.any(vaf < 0):
        max_likelihood = -np.inf
    else:
        max_likelihood = np.sum(n_obs * np.log(e + vaf))
    best_mu = n
    # print(vaf, max_likelihood)
    for m in MASKS:
        mu = np.sum(n_obs[m]) / (1 - e * np.count_nonzero(~m))
        pred_vaf = np.zeros(4)
        pred_vaf[m] = n_obs[m] / mu - e
        likelihood = np.sum(n_obs * np.log(e + pred_vaf))
        if likelihood > max_likelihood and np.all(pred_vaf >= -1e-12):
            max_likelihood = likelihood
            vaf = pred_vaf
            best_mu = mu
            # print(vaf, max_likelihood, best_mu, m)
    return vaf, best_mu


def likelihood_ratio(n_obs, predicted_vaf, ref_allele_index, error_rate):
    if error_rate == 0:
        error_rate += 1e-12
    null_vaf = np.zeros(4)
    null_vaf[ref_allele_index] = 1 - error_rate
    # loglikelihood_null = np.dot(n_obs, np.log(error_rate / 4 + null_vaf))
    # loglikelihood_alt = np.dot(n_obs, np.log(error_rate / 4 + predicted_vaf))
    # test_statistics = 2 * (loglikelihood_alt - loglikelihood_null)
    test_statistics = 2 * np.sum(n_obs * np.log(
        (predicted_vaf + error_rate / 4) / (null_vaf + error_rate / 4)
    ))
    p_val = 1 - chi2(3).cdf(test_statistics)
    return test_statistics, p_val


def f(mu, n, e):
    return np.sum(np.sqrt(1 + (8 / e) ** 2 * (n / mu))) - (8 / e) + 4


def fprime(mu, n, e):
    return (-0.5) * (8 / e) ** 2 / mu * np.sum(n / np.sqrt(1 + (8 / e) ** 2 * n / mu))


def bayesian_factor(n_obs, ref_allele_index, error_rate):
    n = np.sum(n_obs)
    n_ref = n_obs[ref_allele_index]
    p_alt = np.prod(gamma(n_obs)) / gamma(np.sum(n))
    p_null = (1 - error_rate) ** n_ref * error_rate ** (n - n_ref)
    return p_alt / p_null


# def newton_raphson_maximisation(samples, eps=1e-6, damping=1e-3, max_num_iter=1000):
#     num_sample = samples.shape[0]
#     num_category = samples.shape[1]
#     num_trial = 10000
#     samples = np.array(samples)
#     count = np.sum(samples, axis=1).reshape(-1, 1)
#     prob_pred = samples / count
#     p = np.mean(prob_pred, axis=0)
#     v = np.var(prob_pred, axis=0)
#     alpha_pred = p * np.exp(np.mean(np.log(p * (1 - p) / v - 1)[:-1]))
#     for i in range(max_num_iter):
#         q = np.sum(polygamma(1, samples + alpha_pred) - polygamma(1, alpha_pred), axis=0) ** (-1)
#         a = np.sum(alpha_pred)
#         z = np.sum(polygamma(1, a) - polygamma(1, count + a))
#         const = digamma(a) - digamma(count + a)
#         g = np.sum(digamma(samples + alpha_pred) - digamma(alpha_pred) + const.reshape(-1, 1), axis=0)
#         res = np.max(np.abs(g))
#         if res < eps:
#             break
#
#         denominator = 1 / z + np.sum(q)
#         numerator = np.dot(q, g)
#         update_term = q * (g - numerator / denominator)
#         alpha_pred -= damping * update_term
#     return
