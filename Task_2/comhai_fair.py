import numpy as np
from scipy.special import softmax

# comhai and placo for binary tasks
# includes fairness stuff too

# temperature scaling for calibration
def calibrate_temperature(logits, y_true, iters=200, lr=0.05):
    log_T = 0.5
    n = len(y_true)
    for _ in range(iters):
        T = np.exp(log_T)
        probs = softmax(logits / T, axis=1)
        # gradient descent to find best T
        grad = -np.mean(np.sum(probs * (logits/T), axis=1) -
                        logits[np.arange(n), y_true] / T)
        log_T -= lr * (grad - (log_T - 0.5) / (0.25 * n))
    return float(np.exp(log_T))

def apply_calibration(logits, T):
    return softmax(logits / T, axis=1)

# combining model and human
def comhai_combine(m_probs, h_labels, phis):
    # standard comhai rule
    combined = m_probs.copy()
    for lbl, phi in zip(h_labels, phis):
        combined *= phi[int(lbl), :]
        combined = np.clip(combined, 1e-12, None)
    combined /= combined.sum()
    return int(np.argmax(combined))

# greedy selection of humans
def greedy_subset(h_labels, phis, K=2):
    n = len(h_labels)
    M = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            p = np.clip(phis[i][int(h_labels[i]), j], 1e-9, 1-1e-9)
            M[i, j] = p / (1 - p)
    
    C = np.ones(K)
    for j in range(K):
        for i in range(n):
            if M[i, j] > 1.0:
                C[j] *= M[i, j]
                
    j_star = int(np.argmax(C))
    subset = [idx for idx in range(n) if M[idx, j_star] > 1.0]
    return subset if subset else list(range(n))

# estimating labels for placo
def placo_estimate(m_probs, phis):
    est = np.zeros(len(phis), dtype=int)
    for i, phi in enumerate(phis):
        # phi @ m_probs gives expected scores
        est[i] = int(np.argmax(phi @ m_probs))
    return est

# placo prediction with budget
def placo_predict(m_probs, phis, costs, budget, K=2, real_lbls=None):
    est = placo_estimate(m_probs, phis)
    subset_all = greedy_subset(est, phis, K)

    def ratio(i):
        p = np.clip(phis[i][int(est[i]), int(est[i])], 1e-9, 1-1e-9)
        return (p / (1-p)) / max(costs[i], 0.01)

    # sort by value for money
    candidates = sorted(subset_all, key=ratio, reverse=True)
    rem = budget
    picked = []
    for i in candidates:
        if costs[i] <= rem:
            picked.append(i)
            rem -= costs[i]

    if not picked:
        return int(np.argmax(m_probs))

    # use actual labels if we have them (for evaluation)
    labels_to_use = real_lbls[picked] if real_lbls is not None else est[picked]
    return comhai_combine(m_probs, labels_to_use, [phis[i] for i in picked])

# helper to run everything
def evaluate_all(p_test, h_test, phis, y_test, s_test, x_test, costs=None, budget=None, K=2):
    from fairness_metrics import fairness_report
    N = len(y_test)
    num_h = h_test.shape[1]

    results = {}
    if costs is None: costs = np.ones(num_h)
    if budget is None: budget = costs.sum()

    for method in ['model_only', 'best_human', 'all_humans', 'pseudo_lb', 'placo']:
        preds = np.zeros(N, dtype=int)
        for i in range(N):
            mp = p_test[i]
            hl = h_test[i]

            if method == 'model_only':
                preds[i] = int(np.argmax(mp))
            elif method == 'best_human':
                # find human with best avg accuracy
                accs = [np.mean(np.diag(phi)) for phi in phis]
                bst = int(np.argmax(accs))
                preds[i] = comhai_combine(mp, [hl[bst]], [phis[bst]])
            elif method == 'all_humans':
                preds[i] = comhai_combine(mp, hl, phis)
            elif method == 'pseudo_lb':
                sub = greedy_subset(hl, phis, K)
                preds[i] = comhai_combine(mp, hl[sub], [phis[s] for s in sub])
            elif method == 'placo':
                preds[i] = placo_predict(mp, phis, costs, budget, K, real_lbls=hl)

        results[method] = {
            'predictions': preds,
            'metrics': fairness_report(y_test, preds, s_test, x=x_test, label=method)
        }

    return results
