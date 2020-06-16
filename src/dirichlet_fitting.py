import numpy as np
from scipy.special import gamma, loggamma, digamma, polygamma
from scipy.optimize import root_scalar, newton, fsolve, root

import time


def trigamma(x):
    return polygamma(1, x)


def newton_iter(n_obs, eps=1e-8, max_num_iter=200):
    n = np.sum(n_obs)
    alpha = np.array(n_obs, dtype=np.float64)  # np.ones(len(n_obs))
    for i in range(max_num_iter):
        q = (polygamma(1, n_obs + alpha) - polygamma(1, alpha)) ** (-1)
        a = np.sum(alpha)
        z = polygamma(1, a) - polygamma(1, n + a)
        g = digamma(a) - digamma(n + a) + digamma(n_obs + alpha) - digamma(alpha)

        denominator = 1 / z + np.sum(q)
        numerator = np.dot(q, g)
        update_term = q * (g - numerator / denominator)
        alpha -= update_term

        res = np.linalg.norm(g)  # np.linalg.norm(update_term)
        if i % 1 == 0:
            print(alpha, res)
            # print(q, numerator, denominator)
        if res < eps:
            break
    return alpha, res


def func_grad(alpha, n_obs, n):
    a = np.sum(alpha)
    return digamma(a) - digamma(n + a) + digamma(n_obs + alpha) - digamma(alpha)


def func_hess(alpha, n_obs, n):
    a = np.sum(alpha)
    q = polygamma(1, n_obs + alpha) - polygamma(1, alpha)
    z = polygamma(1, a) - polygamma(1, n + a)
    return np.diag(q) + np.ones((4, 4)) * z


def scipy_newton(n_obs, eps=1e-6, max_num_iter=50):
    alpha_init = n_obs
    n = np.sum(n_obs)
    return root(func_grad,
                alpha_init,
                args=(n_obs, n),
                # jac=func_hess,
                tol=eps)


def pseudocount_estimate(df_control,
                         eps=1e-6,
                         damping=1e-2,
                         max_num_iter=10000,
                         polling_freq=1000):
    df = df_control[df_control.sum(axis=1) > 0] + 1  # add-one smoothing and ignore loci with zero coverage.

    result = {}
    start_time = time.time()
    loci = list(set(df.index))
    num_locus = len(loci)
    for k, pos in enumerate(loci):
        samples = np.array(df.loc[pos])
        if samples.ndim != 2 or samples.shape[0] <= 1:
            print(f"{k}: Locus {pos} does not have sufficient samples: {samples}.")
            continue
        if k % polling_freq == 0:
            print(f"Processing position {k} of {num_locus} (loci: {pos}). Num samples: {len(samples)}. "
                  f"Time elapsed: {np.around(time.time() - start_time)} s")

        count = np.sum(samples, axis=1).reshape(-1, 1)
        prob_pred = samples / count
        p = np.mean(prob_pred, axis=0)
        v = np.var(prob_pred, axis=0) + eps**2
        alpha_pred = p * np.exp(np.mean(np.log(p * (1 - p) / v - 1)[:-1]))

        for i in range(max_num_iter):
            q = np.sum(polygamma(1, samples + alpha_pred) - polygamma(1, alpha_pred), axis=0) ** (-1)
            a = np.sum(alpha_pred)
            z = np.sum(polygamma(1, a) - polygamma(1, count + a))
            const = digamma(a) - digamma(count + a)
            g = np.sum(digamma(samples + alpha_pred) - digamma(alpha_pred) + const, axis=0)

            res = np.linalg.norm(np.abs(g))
            if res < eps:
                result[pos] = [i, res]
                result[pos].extend(alpha_pred)
                break
            denominator = 1 / z + np.sum(q)
            numerator = np.dot(q, g)
            update_term = q * (g - numerator / denominator)
            alpha_pred -= damping * update_term
            if i > max_num_iter - 1:
                print(f"{k}: {pos} did not converge. Sample: {samples}")
    return result
