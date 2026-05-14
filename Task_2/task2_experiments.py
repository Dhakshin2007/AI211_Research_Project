"""
Experiments:
  E1: Unbiased humans :- does ComHAI affect fairness?
  E2: Accuracy-biased humans :- mild bias
  E3: Label-biased humans :- moderate bias
  E4: Stereotyping humans :- severe bias
  E5: Mixed pool :- some unbiased, some biased
  E6: Bias progression :- how fairness degrades as bias increases
  E7: PLACO budget vs fairness trade-off
"""

import numpy as np
import sys


from biased_humans  import (generate_biased_dataset, split_dataset,
                             simulate_biased_model_logits,
                             simulate_human_pool,
                             estimate_confusion_binary)
from comhai_fair    import (calibrate_temperature, apply_calibration,
                             evaluate_all)
from fairness_metrics import fairness_report

N_TOTAL   = 8000
N_HUMANS  = 5
K         = 2
BASE_ACC  = 0.75
MIN_DROP  = 0.15
SEED      = 42


def run_one_experiment(bias_level: str, seed: int = 42,
                       n_humans: int = N_HUMANS) -> dict:
    """Full pipeline for one bias level."""
    # Data
    feats, labels, sens = generate_biased_dataset(N_TOTAL, K, seed=seed)
    (f_tr, l_tr, s_tr,
     f_te, l_te, s_te) = split_dataset(feats, labels, sens, seed=seed)

    # AI model logits
    logits_tr = simulate_biased_model_logits(
        f_tr, l_tr, s_tr, BASE_ACC, MIN_DROP, seed=seed)
    logits_te = simulate_biased_model_logits(
        f_te, l_te, s_te, BASE_ACC, MIN_DROP, seed=seed+1)

    # Calibrate
    T = calibrate_temperature(logits_tr, l_tr)
    probs_tr = apply_calibration(logits_tr, T)
    probs_te = apply_calibration(logits_te, T)

    # Human labels
    hl_tr = simulate_human_pool(l_tr, s_tr, n_humans, bias_level, seed=seed)
    hl_te = simulate_human_pool(l_te, s_te, n_humans, bias_level, seed=seed+50)

    # Confusion matrices
    phis = [estimate_confusion_binary(hl_tr[:, i], l_tr)
            for i in range(n_humans)]

    # Costs for PLACO
    costs = np.array([1.0, 1.5, 2.0, 1.0, 2.5])[:n_humans]
    budget = costs.sum() * 0.5  # 50% budget

    # Evaluate all methods
    results = evaluate_all(probs_te, hl_te, phis, l_te, s_te, f_te,
                           costs=costs, budget=budget, K=K)
    return results


def extract_row(results: dict, method: str) -> dict:
    m = results[method]['metrics']
    return {
        'accuracy':  round(m['accuracy']*100, 2),
        'dp_gap':    round(m['dp_gap']*100, 2),
        'eo_gap':    round(m['eo_gap']*100, 2),
        'tpr_gap':   round(m['tpr_gap']*100, 2),
        'fpr_gap':   round(m['fpr_gap']*100, 2),
        'acc_gap':   round(m['acc_gap']*100, 2),
        'if_score':  round(m['if_score']*100, 2) if m['if_score'] else None,
    }


if __name__ == '__main__':
    METHODS = ['model_only', 'best_human', 'all_humans', 'pseudo_lb', 'placo']
    METHOD_NAMES = {
        'model_only':  'AI Only',
        'best_human':  'Best Human',
        'all_humans':  'All Humans',
        'pseudo_lb':   'ComHAI (Greedy)',
        'placo':       'PLACO',
    }
    BIAS_LEVELS = ['none', 'mild', 'moderate', 'severe', 'mixed']

    
    print('  TASK 2 :- FAIRNESS EVALUATION OF HUMAN-AI TEAMS')
    

    #  E1-E5: One table per bias level 
    all_results = {}
    for bias in BIAS_LEVELS:
        print(f'\n[Experiment] Bias level: {bias.upper()}')
        res = run_one_experiment(bias, seed=SEED)
        all_results[bias] = res

        print(f'  {"Method":<22} {"Acc%":>6} {"DP gap":>8} '
              f'{"EO gap":>8} {"TPR gap":>8} {"Acc gap":>8} {"IF%":>6}')
        print('  ' + '-'*70)
        for m in METHODS:
            r = extract_row(res, m)
            print(f'  {METHOD_NAMES[m]:<22} '
                  f'{r["accuracy"]:>6.1f} '
                  f'{r["dp_gap"]:>8.1f} '
                  f'{r["eo_gap"]:>8.1f} '
                  f'{r["tpr_gap"]:>8.1f} '
                  f'{r["acc_gap"]:>8.1f} '
                  f'{str(r["if_score"]):>6}')

    #  E6: Bias progression 
    print('\n' + '='*72)
    print('  E6 :- FAIRNESS DEGRADATION AS HUMAN BIAS INCREASES')
    
    progression = {}
    for bias in BIAS_LEVELS:
        r = extract_row(all_results[bias], 'pseudo_lb')
        progression[bias] = r
        print(f'  {bias:<12}: Acc={r["accuracy"]:.1f}%  '
              f'DP gap={r["dp_gap"]:.1f}%  EO gap={r["eo_gap"]:.1f}%  '
              f'Acc gap={r["acc_gap"]:.1f}%')

    #  E7: Group-level accuracy breakdown 
    print('\n' + '='*72)
    print('  E7 :- PER-GROUP ACCURACY (Majority A=0 vs Minority A=1)')
    
    res_none = all_results['none']
    res_bias = all_results['severe']
    for bias_lbl, res in [('Unbiased humans', res_none),
                           ('Severe bias humans', res_bias)]:
        print(f'\n  {bias_lbl}:')
        print(f'  {"Method":<22} {"Majority A=0":>14} {"Minority A=1":>14} '
              f'{"Gap":>8}')
        print('  ' + '-'*60)
        for m in METHODS:
            abg = res[m]['metrics']['acc_by_group']
            a0  = abg.get(0, 0) * 100
            a1  = abg.get(1, 0) * 100
            gap = abs(a0 - a1)
            print(f'  {METHOD_NAMES[m]:<22} {a0:>14.1f}% {a1:>14.1f}% '
                  f'{gap:>7.1f}%')

    print('\n[DONE] Task 2 experiments complete.')