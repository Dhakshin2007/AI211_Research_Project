"""

 MAIN EXPERIMENT RUNNER
 Reproduces Paper 1 (ComHAI) results + PLACO + Extra Models

"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude/comhai_placo')

from core    import (estimate_confusion_matrix, calibrate_temperature,
                     apply_calibration, predict_instance, placo_predict)
from simulate import (simulate_model_logits, simulate_all_humans,
                      generate_dataset)


# 
# HELPERS
# 

def evaluate_method(method: str,
                    model_probs_test: np.ndarray,
                    human_labels_test: np.ndarray,
                    phis: list,
                    true_labels_test: np.ndarray,
                    K: int,
                    costs: np.ndarray = None,
                    budget: float = None) -> float:
    """Run a method on all test instances and return accuracy."""
    correct = 0
    N = len(true_labels_test)
    n = human_labels_test.shape[1]

    for i in range(N):
        mp = model_probs_test[i]
        hl = human_labels_test[i]

        if method == "placo":
            pred = placo_predict(mp, phis, costs, budget, K,
                                 actual_labels=hl)
        else:
            pred = predict_instance(mp, hl, phis, method=method)

        if pred == true_labels_test[i]:
            correct += 1
    return correct / N * 100.0


def best_human_accuracy(human_labels_test, true_labels_test):
    """Accuracy of the most accurate individual human."""
    n = human_labels_test.shape[1]
    accs = [(human_labels_test[:, i] == true_labels_test).mean() * 100
            for i in range(n)]
    return max(accs)


# 
# EXPERIMENT 1: Reproduce Paper 1 — varying humans & model
# 

def run_experiment(scenario_name: str,
                   human_accuracies: list,
                   model_accuracy: float,
                   K: int = 10,
                   N_train: int = 5000,
                   N_test:  int = 10000,
                   n_runs:  int = 5,
                   seed:    int = 42) -> dict:

    results = {m: [] for m in ['model_only', 'best_human', 'all_humans',
                                'random_subset', 'pseudo_lb']}
    np.random.seed(seed)

    for run in range(n_runs):
        run_seed = seed + run * 100

        # Generate data
        y_train, y_test = generate_dataset(N_train, N_test, K, seed=run_seed)

        # Simulate model logits
        logits_train = simulate_model_logits(y_train, K, model_accuracy,
                                             seed=run_seed)
        logits_test  = simulate_model_logits(y_test,  K, model_accuracy,
                                             seed=run_seed + 1)

        # Calibrate temperature on train set
        T = calibrate_temperature(logits_train, y_train)
        probs_train = apply_calibration(logits_train, T)
        probs_test  = apply_calibration(logits_test,  T)

        # Simulate human labels
        hl_train = simulate_all_humans(y_train, K, human_accuracies,
                                       seed=run_seed)
        hl_test  = simulate_all_humans(y_test,  K, human_accuracies,
                                       seed=run_seed + 50)

        # Estimate confusion matrices from training data
        phis = [estimate_confusion_matrix(hl_train[:, i], y_train, K)
                for i in range(len(human_accuracies))]

        # Evaluate all methods
        for method in results:
            acc = evaluate_method(method, probs_test, hl_test,
                                  phis, y_test, K)
            results[method].append(acc)

    # Average over runs
    return {m: (np.mean(v), np.std(v)) for m, v in results.items()}


# 
# EXPERIMENT 2: Multiple AI models (Paper 1 extension)
# 

AI_MODELS = {
    "Weak CNN (≈57%)":       0.57,
    "Logistic Reg (≈65%)":   0.65,
    "Random Forest (≈75%)":  0.75,
    "SVM (≈80%)":            0.80,
    "ResNet-110 (≈94%)":     0.94,
    "XGBoost (≈85%)":        0.85,
    "KNN (≈60%)":            0.60,
}

# 
# EXPERIMENT 3: PLACO — cost-aware selection
# 

def run_placo_experiment(human_accuracies: list,
                          costs: np.ndarray,
                          budget: float,
                          model_accuracy: float = 0.57,
                          K: int = 10,
                          N_train: int = 5000,
                          N_test:  int = 10000,
                          n_runs:  int = 5,
                          seed:    int = 42) -> dict:

    results = {m: [] for m in ['model_only', 'pseudo_lb', 'placo']}
    cost_savings = []
    np.random.seed(seed)

    for run in range(n_runs):
        run_seed = seed + run * 100
        y_train, y_test = generate_dataset(N_train, N_test, K, seed=run_seed)

        logits_train = simulate_model_logits(y_train, K, model_accuracy,
                                             seed=run_seed)
        logits_test  = simulate_model_logits(y_test,  K, model_accuracy,
                                             seed=run_seed + 1)
        T = calibrate_temperature(logits_train, y_train)
        probs_train = apply_calibration(logits_train, T)
        probs_test  = apply_calibration(logits_test,  T)

        hl_train = simulate_all_humans(y_train, K, human_accuracies,
                                       seed=run_seed)
        hl_test  = simulate_all_humans(y_test,  K, human_accuracies,
                                       seed=run_seed + 50)
        phis = [estimate_confusion_matrix(hl_train[:, i], y_train, K)
                for i in range(len(human_accuracies))]

        for method in ['model_only', 'pseudo_lb']:
            acc = evaluate_method(method, probs_test, hl_test,
                                  phis, y_test, K)
            results[method].append(acc)

        placo_acc = evaluate_method('placo', probs_test, hl_test,
                                    phis, y_test, K, costs=costs,
                                    budget=budget)
        results['placo'].append(placo_acc)
        cost_savings.append(budget / costs.sum() * 100)

    return {m: (np.mean(v), np.std(v)) for m, v in results.items()}, \
           float(np.mean(cost_savings))


# 
# RUN ALL EXPERIMENTS
# 

if __name__ == "__main__":
    print("=" * 65)
    print("  TASK 1 — REPRODUCING PAPER RESULTS")
    print("=" * 65)

    K = 10

    #  Scenario A: 5 humans at 70% accuracy (Paper 1, Row 1, Col 1)
    print("\n[Scenario A] 5 humans @ 70% accuracy | Weak CNN (~57%)")
    r = run_experiment("A", [0.70]*5, model_accuracy=0.57, K=K)
    for m, (mean, std) in r.items():
        print(f"  {m:<20}: {mean:.2f}% ± {std:.2f}%")

    #  Scenario B: 10 humans at 70% accuracy
    print("\n[Scenario B] 10 humans @ 70% accuracy | Weak CNN (~57%)")
    r = run_experiment("B", [0.70]*10, model_accuracy=0.57, K=K)
    for m, (mean, std) in r.items():
        print(f"  {m:<20}: {mean:.2f}% ± {std:.2f}%")

    #  Scenario C: 7 humans 50-80% accuracy (Paper 1 key result)
    print("\n[Scenario C] 7 humans @ 50-80% accuracy | Weak CNN (~57%)")
    r = run_experiment("C", [0.5,0.55,0.6,0.65,0.7,0.75,0.8],
                       model_accuracy=0.57, K=K)
    for m, (mean, std) in r.items():
        print(f"  {m:<20}: {mean:.2f}% ± {std:.2f}%")

    #  Scenario D: ResNet-level model
    print("\n[Scenario D] 7 humans @ 50-80% accuracy | ResNet (~94%)")
    r = run_experiment("D", [0.5,0.55,0.6,0.65,0.7,0.75,0.8],
                       model_accuracy=0.94, K=K)
    for m, (mean, std) in r.items():
        print(f"  {m:<20}: {mean:.2f}% ± {std:.2f}%")

    #  EXTRA MODELS: not in paper
    print("\n" + "=" * 65)
    print("  EXTENDED EVALUATION — MULTIPLE AI BASE MODELS")
    print("=" * 65)
    human_accs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    rows = []
    for model_name, model_acc in AI_MODELS.items():
        r = run_experiment(model_name, human_accs, model_acc, K=K,
                           n_runs=3, seed=99)
        model_only = r['model_only'][0]
        pseudo_lb  = r['pseudo_lb'][0]
        improvement = pseudo_lb - model_only
        rows.append((model_name, model_acc*100, model_only,
                     pseudo_lb, improvement))
        print(f"  {model_name:<28}: AI={model_only:.1f}%  "
              f"ComHAI={pseudo_lb:.1f}%  "
              f"Δ={improvement:+.1f}%")

    #  PLACO Experiment
    print("\n" + "=" * 65)
    print("  PLACO — COST-AWARE SUBSET SELECTION (Paper 2)")
    print("=" * 65)
    costs  = np.array([1.0, 1.5, 2.0, 3.0, 2.5, 1.0, 4.0])   # per-human cost
    budget = 6.0                                                  # total budget
    print(f"  Human costs: {costs}  |  Budget: {budget}  "
          f"|  Total cost if all queried: {costs.sum():.1f}")

    placo_r, saving_pct = run_placo_experiment(
        human_accs, costs, budget, model_accuracy=0.57, K=K)
    print(f"  Model only   : {placo_r['model_only'][0]:.2f}%")
    print(f"  ComHAI (all) : {placo_r['pseudo_lb'][0]:.2f}%")
    print(f"  PLACO        : {placo_r['placo'][0]:.2f}%")
    print(f"  Cost used    : {budget:.1f}/{costs.sum():.1f} "
          f"({saving_pct:.1f}% of total budget)")

    print("\n[DONE] All experiments completed.")
