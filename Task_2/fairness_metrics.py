import numpy as np

# this file has all the fairness formulas from the paper
# we need DP, EO, EqOpp, etc.

# Demographic Parity: check if positive rates are same for everyone
def demographic_parity(y_hat, sens):
    # P(ŷ=1 | A=0) == P(ŷ=1 | A=1)
    grps = np.unique(sens)
    rates = {}
    for g in grps:
        m = sens == g
        rates[int(g)] = np.mean(y_hat[m])

    if len(grps) == 2:
        gap = abs(rates[int(grps[1])] - rates[int(grps[0])])
    else:
        gap = max(rates.values()) - min(rates.values())

    return {
        'positive_rates': rates,
        'dp_gap': gap,
        'satisfied': gap < 0.05 # threshold from class
    }

# Equalized Odds: TPR and FPR should be same
def equalized_odds(y_real, y_hat, sens):
    # EO_gap = max(|TPR_diff|, |FPR_diff|)
    grps = np.unique(sens)
    tpr, fpr = {}, {}
    for g in grps:
        mask = sens == g
        y_r = y_real[mask]
        y_h = y_hat[mask]
        
        pos = y_r == 1
        neg = y_r == 0
        
        tpr[int(g)] = np.mean(y_h[pos]) if pos.sum() > 0 else 0.0
        fpr[int(g)] = np.mean(y_h[neg]) if neg.sum() > 0 else 0.0

    if len(grps) == 2:
        tpr_gap = abs(tpr[int(grps[1])] - tpr[int(grps[0])])
        fpr_gap = abs(fpr[int(grps[1])] - fpr[int(grps[0])])
    else:
        tpr_gap = max(tpr.values()) - min(tpr.values())
        fpr_gap = max(fpr.values()) - min(fpr.values())
        
    eo_gap = max(tpr_gap, fpr_gap)
    return {
        'tpr': tpr, 'fpr': fpr,
        'tpr_gap': tpr_gap, 'fpr_gap': fpr_gap,
        'eo_gap': eo_gap,
        'satisfied': eo_gap < 0.05
    }

# Equal Opportunity: just check TPR
def equal_opportunity(y_real, y_hat, sens):
    grps = np.unique(sens)
    tpr = {}
    for g in grps:
        # only look at cases where true label is 1
        mask = (sens == g) & (y_real == 1)
        tpr[int(g)] = np.mean(y_hat[mask]) if mask.sum() > 0 else 0.0

    if len(grps) == 2:
        gap = abs(tpr[int(grps[1])] - tpr[int(grps[0])])
    else:
        gap = max(tpr.values()) - min(tpr.values())

    return {'tpr': tpr, 'eqopp_gap': gap, 'satisfied': gap < 0.05}

# Predictive Parity: check precision
def predictive_parity(y_real, y_hat, sens):
    grps = np.unique(sens)
    prec = {}
    for g in grps:
        # only look at cases where we predicted 1
        mask = (sens == g) & (y_hat == 1)
        if mask.sum() > 0:
            prec[int(g)] = np.mean(y_real[mask])
        else:
            prec[int(g)] = 0.0

    if len(grps) == 2:
        gap = abs(prec[int(grps[1])] - prec[int(grps[0])])
    else:
        gap = max(prec.values()) - min(prec.values())

    return {'precision': prec, 'pp_gap': gap, 'satisfied': gap < 0.05}

# Individual Fairness: similar people should get similar predictions
def individual_fairness(x, y_hat, pairs=2000, thresh=0.3, seed=42):
    rng = np.random.RandomState(seed)
    N = len(x)
    violations = 0
    sim_pairs = 0

    # normalize so distance makes sense
    x_norm = x / (x.max(axis=0) + 1e-9)

    indices = rng.choice(N, size=(pairs, 2), replace=True)
    for i, j in indices:
        if i == j: continue
        dist = np.linalg.norm(x_norm[i] - x_norm[j])
        if dist < thresh:
            sim_pairs += 1
            if y_hat[i] != y_hat[j]:
                violations += 1

    if sim_pairs == 0:
        return {'if_score': 1.0, 'violations': 0, 'satisfied': True}

    score = 1.0 - violations / sim_pairs
    return {
        'if_score': score,
        'violations': violations,
        'satisfied': score > 0.80
    }

# accuracy per group
def accuracy_by_group(y_real, y_hat, sens):
    grps = np.unique(sens)
    acc = {}
    for g in grps:
        m = sens == g
        acc[int(g)] = np.mean(y_hat[m] == y_real[m])

    if len(grps) == 2:
        gap = abs(acc[int(grps[1])] - acc[int(grps[0])])
    else:
        gap = max(acc.values()) - min(acc.values())

    return {'accuracy': acc, 'acc_gap': gap, 'satisfied': gap < 0.05}

# combined report
def fairness_report(y_real, y_hat, sens, x=None, label=''):
    # this part was tricky to get all in one dict
    acc = np.mean(y_hat == y_real)
    dp = demographic_parity(y_hat, sens)
    eo = equalized_odds(y_real, y_hat, sens)
    eqo = equal_opportunity(y_real, y_hat, sens)
    pp = predictive_parity(y_real, y_hat, sens)
    abg = accuracy_by_group(y_real, y_hat, sens)
    
    if x is not None:
        iff = individual_fairness(x, y_hat)
    else:
        iff = {'if_score': None, 'satisfied': None}

    return {
        'label':         label,
        'accuracy':      acc,
        'dp_gap':        dp['dp_gap'],
        'eo_gap':        eo['eo_gap'],
        'tpr_gap':       eo['tpr_gap'],
        'fpr_gap':       eo['fpr_gap'],
        'eqopp_gap':     eqo['eqopp_gap'],
        'pp_gap':        pp['pp_gap'],
        'acc_gap':       abg['acc_gap'],
        'acc_by_group':  abg['accuracy'],
        'tpr_by_group':  eo['tpr'],
        'fpr_by_group':  eo['fpr'],
        'if_score':      iff.get('if_score'),
        'dp_satisfied':  dp['satisfied'],
        'eo_satisfied':  eo['satisfied'],
        'if_satisfied':  iff.get('satisfied'),
    }
