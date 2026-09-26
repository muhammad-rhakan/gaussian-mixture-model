import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class CustomGMM:
    def __init__(self, n_components, covariance_type='full', max_iter=250, tol=1e-3, random_state=None, verbose=True):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.means = None
        self.covariance = None
        self.weights = None
        self.responsibilities = None
        self.log_likelihood = None
        self.n_iter = None
        self.converged = None


# 1. Initialization
def parameter_initialization(X, n_components, random_state=None):
    np.random.seed(random_state)
    N, D = X.shape

    indices = np.linspace(0, N-1, n_components, dtype=int)
    means = X[indices].copy()

    covariances = np.array([np.eye(D) for _ in range(n_components)])
    weights = np.ones(n_components) / n_components

    return means, covariances, weights


# 2. Expectation Step
def e_step(X, means, covariances, weights):
    N, D = X.shape
    n_components = len(weights)


    responsibilities = np.zeros((N, n_components))

    for k in range(n_components):
        try:
            responsibilities[:,k] = weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k], allow_singular=True)
        except:
            responsibilities[:,k] = weights[k] * 1e-10


    resp_sum = responsibilities.sum(axis=1, keepdims=True)
    resp_sum[resp_sum == 0] = 1e-10

    responsibilities /= resp_sum

    log_likelihood = np.sum(np.log(resp_sum))

    return responsibilities, log_likelihood


# 3. Maximization Step
def m_step(X, responsibilities, covariance_type='full'):
    N, D = X.shape
    n_components = responsibilities.shape[1]

    Nk = responsibilities.sum(axis=0)   # Size of each component
    # Parameters Update
    """ Weight """
    weights = Nk / N

    """ Means """
    means = np.zeros((n_components, D))
    for k in range(n_components):
        if Nk[k] > 0:
            means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]

    """ Covariances """
    covariances = np.zeros((n_components, D, D))
    for k in range(n_components):
        if Nk[k] > 0:
            diff = X - means[k]
            if covariance_type == 'full':
                covariances[k] = np.dot((responsibilities[:, k:k+1] * diff).T, diff) / Nk[k]
                covariances[k] += 1e-6 * np.eye(D)  # regularization
            elif covariance_type == 'diag':
                cov_diag = np.sum(responsibilities[:, k:k+1] * (diff**2), axis=0) / Nk[k]
                covariances[k] = np.diag(cov_diag + 1e-6)

    return means, covariances, weights


def train(X, n_components, max_iter=100, tol=1e-3, covariance_type='full', random_state=None, verbose=True):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()

    means, covariances, weights = parameter_initialization(X, n_components, random_state)

    log_likelihood = -np.inf
    converged = False

    for iteration in range(max_iter):
        responsibilities, new_log_likelihood = e_step(X, means, covariances, weights)

        if abs(new_log_likelihood - log_likelihood) < tol:
            converged = True
            break
        log_likelihood = new_log_likelihood

        means, covariances, weights = m_step(X, responsibilities, covariance_type)

        if verbose and iteration % 20 == 0:
            print(f"Iteration {iteration}, Log-likelihood: {log_likelihood:.4f}")

    if verbose:
        print(f"Converged after {iteration} iterations" if converged else f"Reached max iterations ({max_iter})")

    return {'means': means,
            'covariances': covariances,
            'weights': weights,
            'responsibilities': responsibilities,
            'log_likelihood': log_likelihood}



def gmm_sample(n_samples, means, covariances, weights):
    n_components = len(weights)
    component_indices = np.random.choice(n_components, size=n_samples, p=weights)

    samples = np.array([np.random.multivariate_normal(means[k], covariances[k]) for k in component_indices])

    return samples, component_indices


def gmm_predict_probability(X, means, covariances, weights):
    N = X.shape[0]
    n_components = len(weights)
    probas = np.zeros((N, n_components))

    for k in range(n_components):
        probas[:, k] = weights[k] * multivariate_normal.pdf(X, means[k], covariances[k], allow_singular=True)
    probas /= probas.sum(axis=1, keepdims=True)

    return probas