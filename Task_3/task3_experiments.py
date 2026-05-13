# task3_experiments.py
# this runs all the experiments for task 3

import numpy as np
import sys
import os

# adding task 2 stuff to path so we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task_2')))

from biased_humans    import (generate_biased_dataset, split_dataset,
                               simulate_biased_model_logits,
                               simulate_human_pool,
                               estimate_confusion_binary,
                               estimate_group_confusion)
from comhai_fair      import (calibrate_temperature, apply_calibration,
                               comhai_combine, greedy_subset, placo_predict)
from fairness_metrics import fairness_report
from task3_algorithms import (faircomhai_single, fair_comhai_combine,
                               fair_placo, bias_aware_comhai, search_lambda)

total_n = 8000
K = 2
seed_val = 42

def setup_data(bias='none', seed=seed_val):
    # generate data
    x, y, a = generate_biased_dataset(total_n, K, seed=seed)
    x_tr, y_tr, a_tr, x_te, y_te, a_te = split_dataset(x, y, a, seed=seed)
    
    # model logits
    logits_tr = simulate_biased_model_logits(x_tr, y_tr, a_tr, 0.75, 0.15, seed=seed)
    logits_te = simulate_biased_model_logits(x_te, y_te, a_te, 0.75, 0.15, seed=seed+1)
    
    # calibration
    T = calibrate_temperature(logits_tr, y_tr)
    probs_tr = apply_calibration(logits_tr, T)
    probs_te = apply_calibration(logits_te, T)
    
    # humans
    h_tr = simulate_human_pool(y_tr, a_tr, 5, bias, seed=seed)
    h_te = simulate_human_pool(y_te, a_te, 5, bias, seed=seed+50)
    
    # confusion matrices
    cms = [estimate_confusion_binary(h_tr[:,i], y_tr) for i in range(5)]
    group_cms = [estimate_group_confusion(h_tr[:,i], y_tr, a_tr) for i in range(5)]
    
    # costs for humans
    costs = np.array([1.0, 1.5, 2.0, 1.0, 2.5])
    return (probs_tr, probs_te, y_tr, y_te, a_tr, a_te, x_te, h_tr, h_te, cms, group_cms, costs)

def get_group_acc(probs, labels, s_attr):
    # helper for accuracy per group
    preds = np.argmax(probs, axis=1)
    accs = {}
    for g in [0, 1]:
        mask = s_attr == g
        if mask.sum() > 0:
            accs[g] = float(np.mean(preds[mask] == labels[mask]))
        else:
            accs[g] = 0.5
    return accs

def run_setting(s_num, bias_level, lam=0.3, seed=seed_val):
    (p_tr, p_te, y_tr, y_te, a_tr, a_te, 
     x_te, h_tr, h_te, cms, g_cms, costs) = setup_data(bias_level, seed)
    
    budget = costs.sum() * 0.5
    N = len(y_te)
    g_acc = get_group_acc(p_tr, y_tr, a_tr)

    methods = {}

    # AI baseline
    methods['AI Only'] = np.argmax(p_te, axis=1)

    # basic ComHAI
    p_comhai = np.zeros(N, dtype=int)
    for i in range(N):
        sub = greedy_subset(h_te[i], cms, K)
        p_comhai[i] = comhai_combine(p_te[i], h_te[i][sub], [cms[s] for s in sub])
    methods['ComHAI'] = p_comhai

    # basic PLACO
    p_placo = np.zeros(N, dtype=int)
    for i in range(N):
        p_placo[i] = placo_predict(p_te[i], cms, costs, budget, K, real_lbls=h_te[i])
    methods['PLACO'] = p_placo

    # fair versions
    if s_num == 1:
        # best human only
        best = int(np.argmax([np.mean(np.diag(c)) for c in cms]))
        p_fair = np.zeros(N, dtype=int)
        for i in range(N):
            p_fair[i] = faircomhai_single(p_te[i], h_te[i][best], cms[best], int(a_te[i]), g_acc, lam, K)
        methods['FairComHAI-Single'] = p_fair

    elif s_num == 2:
        p_fair = np.zeros(N, dtype=int)
        for i in range(N):
            p_fair[i] = fair_placo(p_te[i], cms, costs, budget, int(a_te[i]), g_acc, lam, K, real_lbls=h_te[i])
        methods['FairPLACO-Cost'] = p_fair

    elif s_num == 3:
        p_fair = np.zeros(N, dtype=int)
        for i in range(N):
            p_fair[i] = fair_comhai_combine(p_te[i], h_te[i], cms, int(a_te[i]), g_acc, lam, K)
        methods['FairComHAI-Multi'] = p_fair

    elif s_num == 4:
        # this one is similar to setting 2 but with bias later?
        p_fair = np.zeros(N, dtype=int)
        for i in range(N):
            p_fair[i] = fair_placo(p_te[i], cms, costs, budget, int(a_te[i]), g_acc, lam, K, real_lbls=h_te[i])
        methods['FairPLACO-Full'] = p_fair

    elif s_num == 5:
        p_fair = np.zeros(N, dtype=int)
        for i in range(N):
            p_fair[i] = bias_aware_comhai(p_te[i], h_te[i], cms, g_cms, int(a_te[i]), K)
        methods['BiasAware-ComHAI'] = p_fair

    res = {}
    for name, prds in methods.items():
        res[name] = fairness_report(y_te, prds, a_te, x_te, label=name)
    return res

def print_results(results):
    for name, m in results.items():
        a0 = m['acc_by_group'].get(0, 0) * 100
        a1 = m['acc_by_group'].get(1, 0) * 100
        print(f'  {name:<22} Acc={m["accuracy"]*100:5.1f}% '
              f'DP={m["dp_gap"]*100:5.1f}% '
              f'EO={m["eo_gap"]*100:5.1f}% '
              f'[A=0:{a0:.0f}% A=1:{a1:.0f}%]')

if __name__ == '__main__':
    print("--- Running Task 3 Experiments ---")

    # finding best lambda
    print('\nSearching for best lambda...')
    (p_tr, p_te, y_tr, y_te, a_tr, a_te, x_te, h_tr, h_te, cms, g_cms, costs) = setup_data('none', seed_val)
    
    # use half for validation
    val_n = len(y_tr) // 2
    best_lam = search_lambda(p_tr[:val_n], h_tr[:val_n], cms, y_tr[:val_n], a_tr[:val_n], method='fair_multi')
    print(f'Done. Best lam: {best_lam}')

    settings = [
        (1, 'none',     'S1: Single Human (No Bias)'),
        (2, 'none',     'S2: Multi-Human + Cost (No Bias)'),
        (3, 'none',     'S3: Multi-Human (No Bias)'),
        (4, 'moderate', 'S4: Multi-Human + Cost (Mod Bias)'),
        (5, 'severe',   'S5: BiasAware (Severe Bias)'),
    ]

    for snum, b, lbl in settings:
        print(f'\n>>> {lbl}')
        results = run_setting(snum, b, lam=best_lam)
        print_results(results)

    # sensitivity check
    print('\nChecking how lambda affects results...')
    for l in [0.0, 0.2, 0.5, 1.0]:
        r = run_setting(3, 'none', lam=l)
        m = r.get('FairComHAI-Multi', r.get('ComHAI'))
        print(f'Lam {l:.1f}: Acc={m["accuracy"]*100:.1f}%, EO={m["eo_gap"]*100:.1f}%')

    print('\nAll done!')
