"""
=============================================================
 ComHAI + PLACO — Full Implementation from Papers
 Paper 1: Singh, Jain & Jha (AAMAS 2023)
 Paper 2: PLACO (IOS Press 2024)
=============================================================
"""

import numpy as np
from scipy.special import softmax
from typing import List, Tuple, Optional



# 1.  CONFUSION MATRIX ESTIMATION  (Dirichlet Prior)


def estimate_confusion_matrix(human_labels: np.ndarray,
                               true_labels:  np.ndarray,
                               K: int,
                               beta: float = 1.0,
                               gamma: float = 5.0) -> np.ndarray:
    """
    Bayesian Dirichlet-prior estimate of human confusion matrix.
    phi[s, t] = p(human predicts s | true label is t)
    Shape: (K, K)
    """
    # Prior: gamma on diagonal, beta off-diagonal
    prior = np.full((K, K), beta)
    np.fill_diagonal(prior, gamma)

    counts = np.zeros((K, K))
    for s, t in zip(human_labels, true_labels):
        counts[int(s), int(t)] += 1

    # Posterior = counts + prior, then normalise each column
    posterior = counts + prior
    phi = posterior / posterior.sum(axis=0, keepdims=True)
    return phi  # shape (K, K)



# 2.  TEMPERATURE SCALING  (Bayesian version)


def calibrate_temperature(logits: np.ndarray,
                           true_labels: np.ndarray,
                           n_iter: int = 200,
                           lr: float = 0.05) -> float:
    """
    Learns scalar temperature T via gradient ascent on log-likelihood.
    log T ~ N(0.5, 0.5) prior (MAP estimate).
    Returns scalar temperature T > 0.
    """
    log_T = 0.5  # initialise at prior mean
    n = len(true_labels)
    for _ in range(n_iter):
        T = np.exp(log_T)
        scaled = logits / T
        probs  = softmax(scaled, axis=1)
        # gradient of log-likelihood w.r.t. log_T
        p_true = probs[np.arange(n), true_labels]
        grad_ll = -np.mean(np.sum(probs * (logits / T), axis=1) -
                           logits[np.arange(n), true_labels] / T)
        # gradient of Gaussian prior on log_T  (mu=0.5, sigma=0.5)
        grad_prior = -(log_T - 0.5) / (0.5 ** 2)
        log_T -= lr * (grad_ll - grad_prior / n)
    return float(np.exp(log_T))


def apply_calibration(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(logits / T, axis=1)



# 3.  ComHAI COMBINATION  (Equation 1 of Paper 1)


def comhai_combine(model_probs: np.ndarray,
                   human_labels_subset: np.ndarray,
                   phis_subset: List[np.ndarray]) -> int:
    """
    Combine calibrated model probabilities with a subset of human labels.
    model_probs : (K,)
    human_labels_subset : (|S|,)  integer class labels
    phis_subset : list of (K,K) confusion matrices for each human in S
    Returns: predicted class (int)
    """
    K = len(model_probs)
    combined = model_probs.copy()
    for label, phi in zip(human_labels_subset, phis_subset):
        combined *= phi[int(label), :]   # phi[s, j] = p(human says s | true=j)
    if combined.sum() == 0:
        return int(np.argmax(model_probs))
    combined /= combined.sum()
    return int(np.argmax(combined))



# 4.  GreedySubsetSelection  (Algorithm 1 of Paper 1)


def greedy_subset_selection(human_labels: np.ndarray,
                             phis: List[np.ndarray],
                             K: int) -> List[int]:
    """
    Selects pseudo-optimal subset of humans for one instance.
    human_labels : (n,) predicted labels from all n humans
    phis         : list of n confusion matrices (K×K each)
    Returns: list of selected human indices
    """
    n = len(human_labels)
    # M[i][j] = phi[i][h_i, j] / (1 - phi[i][h_i, j])
    M = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            p = phis[i][int(human_labels[i]), j]
            p = np.clip(p, 1e-9, 1 - 1e-9)
            M[i, j] = p / (1.0 - p)

    # For each class j, accumulate product of ratios > 1
    C = np.ones(K)
    for j in range(K):
        for i in range(n):
            if M[i, j] > 1.0:
                C[j] *= M[i, j]

    # Best class
    j_star = int(np.argmax(C))

    # Subset: humans where ratio for j_star > 1
    subset = [i for i in range(n) if M[i, j_star] > 1.0]
    return subset if subset else list(range(n))   # fallback: use all



# 5.  FULL PREDICTION PIPELINE  (one instance)


def predict_instance(model_probs: np.ndarray,
                     human_labels: np.ndarray,
                     phis: List[np.ndarray],
                     method: str = "pseudo_lb") -> int:
    """
    method: 'pseudo_lb'  | 'all_humans' | 'random_subset' |
            'best_human' | 'model_only'
    """
    K = len(model_probs)
    n = len(human_labels)

    if method == "model_only":
        return int(np.argmax(model_probs))

    if method == "best_human":
        # best human = highest diagonal mean in confusion matrix
        accs = [np.mean(np.diag(phi)) for phi in phis]
        best = int(np.argmax(accs))
        return comhai_combine(model_probs, [human_labels[best]], [phis[best]])

    if method == "all_humans":
        return comhai_combine(model_probs, human_labels, phis)

    if method == "random_subset":
        mask = np.random.rand(n) > 0.5
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.array([np.random.randint(n)])
        return comhai_combine(model_probs,
                              human_labels[idx],
                              [phis[i] for i in idx])

    if method == "pseudo_lb":
        subset = greedy_subset_selection(human_labels, phis, K)
        return comhai_combine(model_probs,
                              human_labels[subset],
                              [phis[i] for i in subset])

    raise ValueError(f"Unknown method: {method}")



# 6.  PLACO — Label Estimation  (Paper 2 Stage 1)


def placo_estimate_labels(model_probs: np.ndarray,
                           phis: List[np.ndarray]) -> np.ndarray:
    """
    Estimate each human's label WITHOUT querying them.
    h_hat_i = argmax_s  sum_j  m_j * phi[i][s,j]
    Returns: (n,) estimated labels
    """
    K = len(model_probs)
    n = len(phis)
    estimated = np.zeros(n, dtype=int)
    for i, phi in enumerate(phis):
        # score for each predicted class s
        scores = phi @ model_probs   # (K,) = sum_j phi[s,j]*m_j for each s
        estimated[i] = int(np.argmax(scores))
    return estimated


def placo_value_function(estimated_labels: np.ndarray,
                          phis: List[np.ndarray],
                          K: int,
                          subset: List[int]) -> float:
    """Value function for a given subset using estimated labels."""
    if not subset:
        return 1.0
    val = 1.0
    # Use pseudo-optimal logic with estimated labels
    M = np.zeros((len(subset), K))
    for idx, i in enumerate(subset):
        for j in range(K):
            p = phis[i][int(estimated_labels[i]), j]
            p = np.clip(p, 1e-9, 1 - 1e-9)
            M[idx, j] = p / (1.0 - p)
    C = np.ones(K)
    for j in range(K):
        for idx in range(len(subset)):
            if M[idx, j] > 1.0:
                C[j] *= M[idx, j]
    return float(np.max(C))


def placo_predict(model_probs: np.ndarray,
                  phis: List[np.ndarray],
                  costs: np.ndarray,
                  budget: float,
                  K: int,
                  actual_labels: np.ndarray = None) -> int:
    """
    PLACO full prediction:
    1. Estimate labels without querying to decide WHICH humans to select
    2. Query only selected humans (within budget)
    3. Combine actual queried labels with ComHAI
    """
    n = len(phis)
    estimated_labels = placo_estimate_labels(model_probs, phis)

    # Subset selection using estimated labels (no cost yet)
    subset_all = greedy_subset_selection(estimated_labels, phis, K)

    # Filter by budget — sort by value contribution per unit cost
    def value_contribution(i):
        p = phis[i][int(estimated_labels[i]), int(estimated_labels[i])]
        p = np.clip(p, 1e-9, 1-1e-9)
        ratio = p / (1 - p)
        return ratio / costs[i] if costs[i] > 0 else 0

    candidates = sorted(subset_all,
                        key=value_contribution, reverse=True)
    remaining = budget
    selected = []
    for i in candidates:
        if costs[i] <= remaining:
            selected.append(i)
            remaining -= costs[i]

    if not selected:
        return int(np.argmax(model_probs))

    # Use actual labels for combination if available, else estimated
    if actual_labels is not None:
        query_labels = actual_labels[selected]
    else:
        query_labels = estimated_labels[selected]

    return comhai_combine(model_probs,
                          query_labels,
                          [phis[i] for i in selected])
