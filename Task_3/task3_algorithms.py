import numpy as np
import sys
import os
from scipy.special import softmax

# this file has the algorithms for task 3
# basically combining humans and AI but making it fair

# FairComHAI-Single: single human + AI
def faircomhai_single(m_probs, h_label, phi, s_attr, g_rates, lam=0.5, K=2):
    # combines one human and AI with fairness
    
    # standard Bayesian way
    combined = m_probs.copy()
    combined *= phi[int(h_label), :]
    combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()

    # fairness part
    if len(g_rates) == 2:
        rate_0 = g_rates.get(0, 0.5)
        rate_1 = g_rates.get(1, 0.5)
        diff = rate_0 - rate_1 # positive = majority gets more

        # if minority and disparity is high, boost it
        if s_attr == 1 and diff > 0.05:
            boost = lam * diff
            combined[1] = min(1.0, combined[1] + boost)
            combined[0] = max(0.0, 1.0 - combined[1])

        # if majority and over-predicting, reduce it a bit
        elif s_attr == 0 and diff > 0.05:
            combined[1] = max(0.0, combined[1] - lam * diff * 0.3)
            combined[0] = max(0.0, 1.0 - combined[1])

    combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()
    return int(np.argmax(combined))

# greedy subset with fairness
def fair_greedy_subset(h_labels, phis, s_attr, g_acc, fair_lam=0.3, K=2):
    # selects humans but keeps it fair
    n = len(h_labels)
    M = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            p = np.clip(phis[i][int(h_labels[i]), j], 1e-9, 1 - 1e-9)
            M[i, j] = p / (1 - p)

    C = np.ones(K)
    for j in range(K):
        for i in range(n):
            if M[i, j] > 1.0:
                C[j] *= M[i, j]

    j_star = int(np.argmax(C))

    # check fairness
    disparity = g_acc.get(0, 0.5) - g_acc.get(1, 0.5)
    subset = []
    for idx in range(n):
        acc = M[idx, j_star]
        if acc <= 1.0:
            continue

        diag_acc = phis[idx][j_star, j_star]

        # higher threshold for minority if disparity is large
        if s_attr == 1 and disparity > 0.10:
            thresh = 1.0 + fair_lam * disparity
            if acc > thresh and diag_acc > 0.55:
                subset.append(idx)
        else:
            subset.append(idx)

    return subset if subset else [int(np.argmax([M[i, j_star] for i in range(n)]))]

def fair_comhai_combine(m_probs, h_labels, phis, s_attr, g_acc, lam=0.3, K=2):
    # full combo for multiple humans
    subset = fair_greedy_subset(h_labels, phis, s_attr, g_acc, lam, K)
    
    combined = m_probs.copy()
    for i in subset:
        combined *= phis[i][int(h_labels[i]), :]
        combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()

    disparity = g_acc.get(0, 0.5) - g_acc.get(1, 0.5)
    if s_attr == 1 and disparity > 0.05:
        boost = lam * 0.5 * disparity
        combined[1] = min(0.95, combined[1] + boost)
        combined[0] = max(0.05, 1.0 - combined[1])

    combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()
    return int(np.argmax(combined))

# FairPLACO-Cost: multiple humans, different costs, fairness + cost
def fair_placo(m_probs, phis, costs, budget, s_attr, g_acc, lam=0.3, K=2, real_lbls=None):
    n = len(phis)
    # estimate labels for free first
    est = np.zeros(n, dtype=int)
    for i, phi in enumerate(phis):
        est[i] = int(np.argmax(phi @ m_probs))

    j_hat = int(np.argmax(m_probs))
    M = np.zeros(n)
    for i in range(n):
        p = np.clip(phis[i][int(est[i]), j_hat], 1e-9, 1 - 1e-9)
        M[i] = p / (1 - p)

    # fairness penalty
    disparity = g_acc.get(0, 0.5) - g_acc.get(1, 0.5)
    penalty = np.zeros(n)
    if s_attr == 1 and disparity > 0.05:
        for i in range(n):
            if K == 2:
                err = phis[i][0, 1]  # error rate
                penalty[i] = lam * err * disparity

    val = M - penalty

    # greedy selection based on budget
    candidates = [(i, val[i] / max(costs[i], 0.01))
                  for i in range(n) if M[i] > 1.0]
    candidates.sort(key=lambda x: x[1], reverse=True)

    left = budget
    picked = []
    for idx, _ in candidates:
        if costs[idx] <= left:
            picked.append(idx)
            left -= costs[idx]

    if not picked:
        return int(np.argmax(m_probs))

    labels_to_use = real_lbls[picked] if real_lbls is not None else est[picked]

    combined = m_probs.copy()
    for i, lbl in zip(picked, labels_to_use):
        combined *= phis[i][int(lbl), :]
        combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()

    if s_attr == 1 and disparity > 0.05:
        boost = lam * 0.4 * disparity
        combined[1] = min(0.95, combined[1] + boost)
        combined[0] = max(0.05, 1.0 - combined[1])

    combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()
    return int(np.argmax(combined))

# de-biasing humans with group specific matrices
def bias_aware_comhai(m_probs, h_labels, phis, phis_group, s_attr, K=2):
    combined = m_probs.copy()
    for i in range(len(h_labels)):
        lbl = int(h_labels[i])
        g = int(s_attr)
        if phis_group and i < len(phis_group) and g in phis_group[i]:
            phi_use = phis_group[i][g]
        else:
            phi_use = phis[i]
        combined *= phi_use[lbl, :]
        combined = np.clip(combined, 1e-12, None)

    combined /= combined.sum()

    # greedy part
    n = len(h_labels)
    M = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            g = int(s_attr)
            phi_use = phis_group[i][g] if (phis_group and i < len(phis_group)
                                           and g in phis_group[i]) else phis[i]
            p = np.clip(phi_use[int(h_labels[i]), j], 1e-9, 1 - 1e-9)
            M[i, j] = p / (1 - p)

    C = np.ones(K)
    for j in range(K):
        for i in range(n):
            if M[i, j] > 1.0:
                C[j] *= M[i, j]
    j_star = int(np.argmax(C))
    subset = [i for i in range(n) if M[i, j_star] > 1.0]
    
    if not subset:
        return int(np.argmax(m_probs))

    # final combo
    combined = m_probs.copy()
    for idx in subset:
        g = int(s_attr)
        phi_use = phis_group[idx][g] if (phis_group and idx < len(phis_group)
                                         and g in phis_group[idx]) else phis[idx]
        combined *= phi_use[int(h_labels[idx]), :]
        combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()
    return int(np.argmax(combined))

# cross validation to find best lam
def search_lambda(m_val, h_val, phis, true_val, s_val, lams=None, method='fair_multi', K=2, costs=None, budget=None):
    if lams is None:
        lams = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    # need task 2 metrics
    t2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task_2'))
    if t2_path not in sys.path:
        sys.path.append(t2_path)
    from fairness_metrics import equalized_odds

    best_lam = 0.3
    best_score = -np.inf
    N = len(true_val)

    for lam in lams:
        preds = np.zeros(N, dtype=int)
        g_acc = {}
        for g in [0, 1]:
            mask = s_val == g
            if mask.sum() > 0:
                g_acc[g] = np.mean(
                    np.array([int(np.argmax(m_val[i]))
                               for i in range(N)])[mask] == true_val[mask])

        for i in range(N):
            mp = m_val[i]
            hl = h_val[i]
            sa = int(s_val[i])
            if method == 'fair_single':
                preds[i] = faircomhai_single(mp, hl[0], phis[0], sa, g_acc, lam, K)
            elif method == 'fair_multi':
                preds[i] = fair_comhai_combine(mp, hl, phis, sa, g_acc, lam, K)
            elif method == 'fair_placo':
                if costs is None: costs = np.ones(len(phis))
                if budget is None: budget = costs.sum() * 0.5
                preds[i] = fair_placo(mp, phis, costs, budget, sa, g_acc, lam, K, real_lbls=hl)

        acc = np.mean(preds == true_val)
        # using eo gap as penalty
        eo = equalized_odds(true_val, preds, s_val)['eo_gap']
        score = acc - 0.5 * eo   # balance accuracy and fairness
        if score > best_score:
            best_score = score
            best_lam = lam

    return best_lam
