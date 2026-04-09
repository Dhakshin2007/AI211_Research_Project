"""

 Data & Model Simulation — mimics CIFAR-10H setup
 from Paper 1 Section 5.1

"""

import numpy as np
from scipy.special import softmax
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1.  SIMULATE AI MODEL LOGITS  (different accuracy levels)
# ─────────────────────────────────────────────────────────────

def simulate_model_logits(true_labels: np.ndarray,
                           K: int,
                           model_accuracy: float,
                           noise_scale: float = 2.0,
                           seed: int = 42) -> np.ndarray:
    """
    Simulate logits for a model with a given accuracy.
    - Correct class gets a signal + noise
    - Wrong classes get noise only
    """
    rng = np.random.RandomState(seed)
    N = len(true_labels)
    logits = rng.randn(N, K) * noise_scale

    # Boost the correct class proportional to accuracy
    signal = noise_scale * (model_accuracy / (1 - model_accuracy + 1e-9)) ** 0.5
    for i, y in enumerate(true_labels):
        logits[i, y] += signal

    return logits


# ─────────────────────────────────────────────────────────────
# 2.  SIMULATE HUMAN LABELS  (Paper 1 Section 5.1 exact method)
# ─────────────────────────────────────────────────────────────

def simulate_human_label(true_label: int,
                          K: int,
                          accuracy: float,
                          confusion_dist: Optional[np.ndarray] = None,
                          rng: np.random.RandomState = None) -> int:
    """
    h^[psi](x):
      - with prob accuracy  → return true_label
      - with prob 1-accuracy → sample wrong class from confusion_dist
    """
    if rng is None:
        rng = np.random.RandomState()
    if rng.rand() < accuracy:
        return true_label
    # Sample from wrong classes
    if confusion_dist is None:
        wrong = [k for k in range(K) if k != true_label]
        return int(rng.choice(wrong))
    # Use provided distribution over wrong classes
    dist = confusion_dist.copy()
    dist[true_label] = 0.0
    if dist.sum() == 0:
        dist = np.ones(K) / K
        dist[true_label] = 0.0
    dist /= dist.sum()
    return int(rng.choice(K, p=dist))


def simulate_all_humans(true_labels: np.ndarray,
                         K: int,
                         human_accuracies: list,
                         seed: int = 0) -> np.ndarray:
    """
    Returns human_labels of shape (N, n_humans)
    """
    rng = np.random.RandomState(seed)
    N = len(true_labels)
    n = len(human_accuracies)
    labels = np.zeros((N, n), dtype=int)
    for i, acc in enumerate(human_accuracies):
        for j, y in enumerate(true_labels):
            labels[j, i] = simulate_human_label(y, K, acc, rng=rng)
    return labels


# ─────────────────────────────────────────────────────────────
# 3.  GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────

def generate_dataset(N_train: int = 5000,
                     N_test:  int = 10000,
                     K:       int = 10,
                     seed:    int = 42):
    """
    Synthetic K-class dataset with uniform class distribution.
    Returns train and test true labels.
    """
    rng = np.random.RandomState(seed)
    y_train = rng.randint(0, K, N_train)
    y_test  = rng.randint(0, K, N_test)
    return y_train, y_test



