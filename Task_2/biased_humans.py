import numpy as np

# this script simulates humans with different bias levels
# we have Level 0 (unbiased) up to Level 3 (stereotyping)

# dataset generator
def generate_biased_dataset(N=10000, K=2, d=8, m_frac=0.30, seed=42):
    # A=0 is majority, A=1 is minority
    rng = np.random.RandomState(seed)

    # sensitive attribute
    s_attr = (rng.rand(N) < m_frac).astype(int)

    # Features - adding some shift so minority looks a bit different
    X = rng.randn(N, d)
    X[s_attr == 1] += 0.3 

    # get true labels with a sigmoid
    y_true = np.zeros(N, dtype=int)
    for i in range(N):
        signal = X[i, :4].mean()
        p_pos = 1 / (1 + np.exp(-signal)) # logistic
        y_true[i] = int(rng.rand() < p_pos)

    return X, y_true, s_attr

def split_dataset(X, y, s, train_frac=0.5, seed=42):
    # standard train/test split
    rng = np.random.RandomState(seed)
    N = len(y)
    idx = rng.permutation(N)
    n_tr = int(N * train_frac)
    tr = idx[:n_tr]; te = idx[n_tr:]
    return X[tr], y[tr], s[tr], X[te], y[te], s[te]

# AI model logits - biased against minority
def simulate_biased_model_logits(X, y_true, s_attr, base_acc=0.75, min_drop=0.15, seed=42):
    rng = np.random.RandomState(seed)
    N = len(y_true)
    K = 2

    logits = np.zeros((N, K))
    for i in range(N):
        # lower accuracy if s_attr is 1
        acc = base_acc if s_attr[i] == 0 else (base_acc - min_drop)
        acc = np.clip(acc, 0.51, 0.99)

        if rng.rand() < acc:
            sig = 1.5 
        else:
            sig = -0.5 

        logits[i, y_true[i]] = sig + rng.randn() * 0.3
        logits[i, 1-y_true[i]] = -sig + rng.randn() * 0.3

    return logits

# Different human types

def simulate_human_unbiased(y_true, s_attr, acc=0.75, seed=0):
    # Level 0 - normal human
    rng = np.random.RandomState(seed)
    N = len(y_true)
    preds = np.zeros(N, dtype=int)
    for i in range(N):
        if rng.rand() < acc:
            preds[i] = y_true[i]
        else:
            preds[i] = 1 - y_true[i]
    return preds

def simulate_human_accuracy_bias(y_true, s_attr, maj_acc=0.80, min_acc=0.60, seed=0):
    # Level 1 - just bad at minority group
    rng = np.random.RandomState(seed)
    N = len(y_true)
    preds = np.zeros(N, dtype=int)
    for i in range(N):
        cur_acc = maj_acc if s_attr[i] == 0 else min_acc
        if rng.rand() < cur_acc:
            preds[i] = y_true[i]
        else:
            preds[i] = 1 - y_true[i]
    return preds

def simulate_human_label_bias(y_true, s_attr, acc=0.75, b_rate=0.25, b_dir=0, seed=0):
    # Level 2 - systematically picking one class for minority
    rng = np.random.RandomState(seed)
    N = len(y_true)
    preds = np.zeros(N, dtype=int)
    for i in range(N):
        if s_attr[i] == 1 and rng.rand() < b_rate:
            preds[i] = b_dir 
        elif rng.rand() < acc:
            preds[i] = y_true[i]
        else:
            preds[i] = 1 - y_true[i]
    return preds

def simulate_human_stereotyping(y_true, s_attr, acc=0.75, s_rate=0.30, seed=0):
    # Level 3 - always picking negative for minority
    rng = np.random.RandomState(seed)
    N = len(y_true)
    preds = np.zeros(N, dtype=int)
    for i in range(N):
        if s_attr[i] == 1 and rng.rand() < s_rate:
            preds[i] = 0 
        elif rng.rand() < acc:
            preds[i] = y_true[i]
        else:
            preds[i] = 1 - y_true[i]
    return preds

# full pool of humans
def simulate_human_pool(y_true, s_attr, n_h=5, bias_type='none', seed=0):
    N = len(y_true)
    labels = np.zeros((N, n_h), dtype=int)

    for i in range(n_h):
        s = seed + i * 17
        acc = 0.70 + i * 0.02 

        if bias_type == 'none':
            labels[:, i] = simulate_human_unbiased(y_true, s_attr, acc, s)
        elif bias_type == 'mild':
            labels[:, i] = simulate_human_accuracy_bias(y_true, s_attr, 0.78, 0.62, s)
        elif bias_type == 'moderate':
            labels[:, i] = simulate_human_label_bias(y_true, s_attr, acc, 0.20, 0, s)
        elif bias_type == 'severe':
            labels[:, i] = simulate_human_stereotyping(y_true, s_attr, acc, 0.35, s)
        elif bias_type == 'mixed':
            # half and half
            if i < n_h // 2:
                labels[:, i] = simulate_human_unbiased(y_true, s_attr, acc, s)
            else:
                labels[:, i] = simulate_human_accuracy_bias(y_true, s_attr, 0.78, 0.58, s)

    return labels

# confusion matrix estimation
def estimate_confusion_binary(h_labels, y_true, K=2, beta=1.0, gamma=5.0):
    # using a dirichlet prior like we saw in class
    prior = np.full((K, K), beta)
    np.fill_diagonal(prior, gamma)
    counts = np.zeros((K, K))
    for s, t in zip(h_labels, y_true):
        counts[int(s), int(t)] += 1
    post = counts + prior
    return post / post.sum(axis=0, keepdims=True)

def estimate_group_confusion(h_labels, y_true, s_attr, K=2):
    # confusion matrix for each group separately
    grps = np.unique(s_attr)
    res = {}
    for g in grps:
        m = s_attr == g
        res[int(g)] = estimate_confusion_binary(h_labels[m], y_true[m], K)
    return res
