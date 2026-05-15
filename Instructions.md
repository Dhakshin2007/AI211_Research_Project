# Fairness in Human-AI Teams — Complete Project Instructions

**Mentor:** Dr. Shweta Jain | **Institute:** IIT Ropar | **Year:** 2024-25  
**Papers:** ComHAI (AAMAS 2023) + PLACO (IOS Press 2024)

---

## 📁 Project File Structure

```
project/
│
├── 📄 INSTRUCTIONS.md              ← You are here
│
├── ── TASK 1 FILES ──────────────────────────────────────────
├── core.py                         ← ComHAI + PLACO algorithms
├── simulate.py                     ← Human annotation simulation
├── experiments.py                  ← Run Task 1 experiments
│
├── ── TASK 2 FILES ──────────────────────────────────────────
├── fairness_metrics.py             ← All 5 fairness metrics
├── biased_humans.py                ← Human bias simulation
├── comhai_fair.py                  ← Binary ComHAI + PLACO
├── task2_experiments.py            ← Run Task 2 experiments
│
├── ── TASK 3 FILES ──────────────────────────────────────────
├── task3_algorithms.py             ← 5 novel fair algorithms
├── task3_experiments.py            ← Run Task 3 experiments
│
└── ── PDF REPORTS ───────────────────────────────────────────
    ├── Task1_ComHAI_PLACO_Report.pdf       ← Academic report (LaTeX)
    ├── Task1_Complete_Study_Guide.pdf      ← Study guide + Q&A
    ├── Task2_Fairness_Study_Guide.pdf      ← Fairness study guide
    ├── Task3_Novel_Algorithms_Study_Guide.pdf  ← Task 3 study guide
    └── Complete_Project_All_Tasks.pdf      ← Master document (all tasks)
```

---

## ⚙️ Requirements

Install once before running anything:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas reportlab
```

All files use only standard scientific Python libraries — no internet required after install.

---

---

# TASK 1 — Survey, Reproduction & Shortcomings

## Files Involved

| File | Role |
|------|------|
| `core.py` | All ComHAI + PLACO algorithm implementations |
| `simulate.py` | Simulates CIFAR-10H human annotation data |
| `experiments.py` | Runs all Task 1 experiments |

---

## How to Run

```bash
python3 experiments.py
```

That's it. No arguments needed. Takes **2–3 minutes** to complete.

---

## What Happens Step by Step

When you run `experiments.py`, it executes 6 experiments in order:

### Step 1 — Setup header prints
```

  TASK 1 — REPRODUCING PAPER RESULTS

```
Nothing is computed yet. Just a visual separator.

---

### Step 2 — Scenario A runs
```
[Scenario A] 5 humans @ 70% accuracy | Weak CNN (~57%)
```
**What it does:**
- Generates 3,000 training instances and 5,000 test instances (10-class synthetic data)
- Simulates a weak CNN model with ~57% accuracy
- Simulates 5 humans each with 70% accuracy
- Estimates each human's 10×10 confusion matrix using Dirichlet prior
- Learns temperature T for model calibration
- Tests all 5 methods on test set, averaged over 3 random seeds

**Output you see:**
```
  model_only          : 38.87% ± 0.57%
  best_human          : 71.72% ± 0.38%
  all_humans          : 96.54% ± 0.15%
  random_subset       : 84.82% ± 0.33%
  pseudo_lb           : 95.15% ± 0.28%
```

**What the numbers mean:**
- `model_only` → AI model predicting alone, no humans → baseline
- `best_human` → Most accurate single human combined with AI
- `all_humans` → All 5 humans blindly combined (non-monotone risk)
- `random_subset` → Random 50% of humans selected each time
- `pseudo_lb` → **ComHAI GreedySubsetSelection** — the paper's method

---

### Step 3 — Scenarios B, C, D run (same format)

```
[Scenario B] 10 humans @ 70% accuracy | Weak CNN (~57%)
[Scenario C] 7 humans @ 50-80% accuracy | Weak CNN (~57%)
[Scenario D] 7 humans @ 50-80% accuracy | ResNet (~94%)
```

**What changes each scenario:**
- **Sc-B:** More humans (10 instead of 5) → all methods improve
- **Sc-C:** Mixed accuracy humans (50%, 55%, 60%, 65%, 70%, 75%, 80%) → realistic
- **Sc-D:** ResNet model (94% alone) → AI is strong so humans add less

---

### Step 4 — Extended evaluation (your original contribution)
```

  EXTENDED EVALUATION — MULTIPLE AI BASE MODELS

  Weak CNN (≈57%)             : AI=39.4%  ComHAI=95.3%  Δ=+55.9%
  KNN (≈60%)                  : AI=41.8%  ComHAI=95.3%  Δ=+53.5%
  Logistic Reg (≈65%)         : AI=46.4%  ComHAI=95.4%  Δ=+49.1%
  Random Forest (≈75%)        : AI=59.0%  ComHAI=96.0%  Δ=+37.1%
  SVM (≈80%)                  : AI=67.4%  ComHAI=96.6%  Δ=+29.2%
  XGBoost (≈85%)              : AI=77.8%  ComHAI=97.5%  Δ=+19.7%
  ResNet-110 (≈94%)           : AI=98.1%  ComHAI=99.6%  Δ=+1.5%
```

**What this means:**
- `AI=` → what that model achieves predicting alone
- `ComHAI=` → what ComHAI achieves combining that model with 7 humans
- `Δ=` → improvement in percentage points (weaker AI = more benefit)
- **KNN, Logistic Reg, XGBoost are NOT in the original paper** — this is your extension

---

### Step 5 — PLACO experiment (Paper 2)
```

  PLACO — COST-AWARE SUBSET SELECTION (Paper 2)

  Human costs: [1.  1.5 2.  3.  2.5 1.  4. ]  |  Budget: 6.0  |  Total: 15.0
  Model only   : 38.87%
  ComHAI (all) : 95.61%
  PLACO        : 88.94%
  Cost used    : 6.0/15.0 (40.0% of total budget)
```

**What this means:**
- 7 humans each have a different cost (like querying a doctor vs a crowd worker)
- Total cost to ask everyone = 15.0 units
- Budget = 6.0 units (40% of total)
- PLACO estimates labels for free first, then selects cheapest useful subset
- Gets 88.94% accuracy spending only 6 out of 15 units
- **Trade-off:** 95.6% with full budget vs 88.9% with 40% budget

---

### Final line
```
[DONE] All experiments completed.
```

---

## Task 1 PDF Reports

| PDF | What it is |
|-----|-----------|
| `Task1_ComHAI_PLACO_Report.pdf` | Formal 4-page academic report with TikZ flowcharts — show this to professor |
| `Task1_Complete_Study_Guide.pdf` | 10-page study guide with concepts, code explanations, Q&A prep |

---

---

# TASK 2 — Fairness Evaluation

## Files Involved

| File | Role |
|------|------|
| `fairness_metrics.py` | 5 fairness metrics implemented from scratch |
| `biased_humans.py` | 4-level human bias simulation engine |
| `comhai_fair.py` | ComHAI + PLACO adapted for binary classification |
| `task2_experiments.py` | Runs all Task 2 experiments |

---

## How to Run

```bash
python3 task2_experiments.py
```

Takes **3–5 minutes** to complete. Must be run from the folder containing all task 2 files.

> **Note:** `task2_experiments.py` imports from `fairness_metrics.py`, `biased_humans.py`, and `comhai_fair.py`. Keep all files in the same folder.

---

## What Happens Step by Step

### Step 1 — Header
```

  TASK 2 — FAIRNESS EVALUATION OF HUMAN-AI TEAMS

```

---

### Step 2 — One block per bias level (5 blocks total)

For each bias level (`none`, `mild`, `moderate`, `severe`, `mixed`), you see:

```
[Experiment] Bias level: NONE

  Method                 Acc%   DP gap   EO gap  TPR gap  Acc gap   IF%
  ----------------------------------------------------------------------
  AI Only              70.4     4.6      17.4     11.5     14.1   100.0
  Best Human           92.2     7.7       6.2      3.4      4.6   100.0
  All Humans           93.7     7.4       3.3      2.0      2.5   100.0
  PLACO                90.5     6.1       3.0      3.0      2.9   100.0
  ComHAI (Greedy)      88.4     5.5       0.9      0.9      0.2   100.0
```

**What each column means:**

| Column | Full Name | Meaning | Good value |
|--------|-----------|---------|------------|
| `Acc%` | Accuracy | Overall correct predictions (%) | Higher |
| `DP gap` | Demographic Parity gap | Difference in positive prediction rates between groups | Lower (< 5%) |
| `EO gap` | Equalized Odds gap | Max of TPR gap and FPR gap across groups | Lower (< 5%) |
| `TPR gap` | True Positive Rate gap | Difference in sensitivity between groups | Lower |
| `Acc gap` | Accuracy gap | Difference in accuracy between majority (A=0) and minority (A=1) | Lower |
| `IF%` | Individual Fairness | % of similar input pairs getting same prediction | Higher (> 80%) |

**What you are looking for:** Methods where `Acc%` stays high AND all gap columns are small. That is a fair AND accurate method.

---

### Step 3 — Bias progression summary (E6)

```

  E6 — FAIRNESS DEGRADATION AS HUMAN BIAS INCREASES

  none        : Acc=88.4%  DP gap=5.5%  EO gap=0.9%  Acc gap=0.2%
  mild        : Acc=86.2%  DP gap=3.7%  EO gap=23.1% Acc gap=22.5%
  moderate    : Acc=85.6%  DP gap=9.6%  EO gap=26.1% Acc gap=14.9%
  severe      : Acc=82.8%  DP gap=18.5% EO gap=41.2% Acc gap=24.2%
  mixed       : Acc=85.5%  DP gap=4.2%  EO gap=16.9% Acc gap=16.5%
```

**What this means:**
- Shows ComHAI's fairness as human bias increases
- `none` → ComHAI is very fair (EO gap only 0.9%)
- `severe` → ComHAI is very unfair (EO gap jumps to 41.2%)
- This is the critical finding: **ComHAI amplifies human bias**

---

### Step 4 — Per-group accuracy breakdown (E7)

```

  E7 — PER-GROUP ACCURACY (Majority A=0 vs Minority A=1)


  Unbiased humans:
  Method                 Majority A=0    Minority A=1     Gap
  ------------------------------------------------------------
  AI Only                      74.7%           60.6%    14.1%
  ComHAI (Greedy)              88.4%           88.2%     0.2%

  Severe bias humans:
  Method                 Majority A=0    Minority A=1     Gap
  ------------------------------------------------------------
  AI Only                      74.7%           60.6%    14.1%
  ComHAI (Greedy)              90.2%           66.0%    24.2%
```

**What this means:**
- `A=0` column → accuracy for majority group
- `A=1` column → accuracy for minority group
- `Gap` → how unfair the method is (0% = perfectly fair)
- With unbiased humans: ComHAI gap = 0.2% (excellent)
- With severe bias: ComHAI gap = 24.2% (very unfair — worse than AI alone)

---

## Task 2 PDF Report

| PDF | What it is |
|-----|-----------|
| `Task2_Fairness_Study_Guide.pdf` | 10-page guide covering all 5 fairness metrics, 4 bias levels, all results, and Q&A |

---

---

# TASK 3 — Novel Fairness-Aware Algorithms

## Files Involved

| File | Role |
|------|------|
| `task3_algorithms.py` | All 5 novel algorithm implementations |
| `task3_experiments.py` | Runs all 5 settings + lambda sensitivity |

> Also needs Task 2 files (`biased_humans.py`, `comhai_fair.py`, `fairness_metrics.py`) in the same folder.

---

## How to Run

```bash
python3 task3_experiments.py
```

Takes **4–6 minutes** to complete.

---

## What Happens Step by Step

### Step 1 — Header
```

  TASK 3 — NOVEL FAIRNESS-AWARE HUMAN-AI ALGORITHMS

```

---

### Step 2 — Lambda search
```
[Lambda Search] Finding optimal fairness weight...
  Best lambda = 0.3
```

**What this means:**
- Lambda (λ) controls the accuracy-fairness trade-off
- The code searches λ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
- Chooses the value that maximises `Accuracy − 0.5 × EO_gap`
- `Best lambda = 0.3` means this balances accuracy and fairness best

---

### Step 3 — All 5 settings run

For each setting you see:

```
[Setting 1: Single Human + AI (Unbiased)]
  AI Only                Acc=70.4%  DP=4.6%   EO=17.4%  AccGap=14.1%  [A=0:75% A=1:61%]
  ComHAI                 Acc=88.4%  DP=5.5%   EO=0.9%   AccGap=0.2%   [A=0:88% A=1:88%]
  PLACO                  Acc=90.5%  DP=6.1%   EO=3.0%   AccGap=2.9%   [A=0:91% A=1:89%]
  FairComHAI-Single      Acc=88.4%  DP=5.5%   EO=0.8%   AccGap=0.2%   [A=0:88% A=1:88%]
```

**What the columns mean:**

| Column | Meaning |
|--------|---------|
| `Acc=` | Overall accuracy — higher is better |
| `DP=` | Demographic Parity gap — how unequal positive prediction rates are |
| `EO=` | Equalized Odds gap — how unequal TPR+FPR are between groups |
| `AccGap=` | Accuracy gap between majority and minority — lower is fairer |
| `[A=0: A=1:]` | Individual accuracy for majority (A=0) and minority (A=1) groups |

**The 5 settings and their novel algorithm:**

| Setting | Novel Algorithm | What it tests |
|---------|----------------|---------------|
| Setting 1 | `FairComHAI-Single` | Single human + AI with fairness correction |
| Setting 2 | `FairPLACO-Cost` | Multi-human + cost + fairness penalty |
| Setting 3 | `FairComHAI-Multi` | Multi-human + AI with strict minority threshold |
| Setting 4 | `FairPLACO-Full` | All objectives: accuracy + cost + fairness |
| Setting 5 | `BiasAware-ComHAI` | Biased humans using group-conditioned confusion matrices |

---

### Step 4 — Lambda sensitivity table

```

  LAMBDA SENSITIVITY: Effect of Fairness Weight

  Lambda       Accuracy     EO Gap     DP Gap    Acc Gap
  -------------------------------------------------------
  0.0           88.4%        0.9%       5.5%       0.2%
  0.1           88.4%        0.9%       5.5%       0.2%
  0.3           88.4%        0.9%       5.5%       0.2%
  0.5           88.4%        0.9%       5.5%       0.2%
  1.0           88.4%        0.8%       5.4%       0.2%
```

**What this means:**
- Shows how changing lambda affects accuracy vs fairness
- With unbiased humans, lambda has little effect because ComHAI is already fair
- Lambda matters most under biased humans (Settings 4 and 5)
- Use this to choose your lambda for a specific scenario

---

### Final line
```
[DONE] All Task 3 experiments complete.
```

---

## Task 3 PDF Reports

| PDF | What it is |
|-----|-----------|
| `Task3_Novel_Algorithms_Study_Guide.pdf` | Deep dive into all 5 algorithms with theory, pseudocode, results and Q&A |
| `Complete_Project_All_Tasks.pdf` | Master document covering all 3 tasks — use this for your professor |

---

---

# 📊 Quick Reference — What Each Output Number Means

## Fairness Metrics at a Glance

| Metric | Formula | What 0% means | What high % means |
|--------|---------|---------------|-------------------|
| DP gap | \|P(ŷ=1\|A=0) − P(ŷ=1\|A=1)\| | Both groups get same positive rate | One group gets far more positives |
| EO gap | max(\|ΔTPR\|, \|ΔFPR\|) | Both groups equally sensitive + specific | One group harder to classify correctly |
| Acc gap | \|Acc(A=0) − Acc(A=1)\| | Both groups equally accurate | Minority group treated worse |
| IF score | % similar pairs with same prediction | — | 100% = perfectly individually fair |

> **Rule of thumb:** Any gap below 5% is considered acceptable. Above 10% is a fairness concern.

---

## Results Interpretation Guide

```
ComHAI (Greedy)   Acc=88.4%   EO=0.9%   AccGap=0.2%   ✅ GOOD
ComHAI (severe)   Acc=82.8%   EO=41.2%  AccGap=24.2%  ❌ UNFAIR
BiasAware         Acc=84.8%   EO=27.6%  AccGap=11.7%  ⚠️ IMPROVED
```

- ✅ High Acc + Low gaps = method works well for everyone
- ❌ High Acc + High gaps = method helps majority but hurts minority
- ⚠️ Middle ground = our novel algorithms moving in the right direction

---

# 📋 Complete Run Order (All Tasks)

Run them in this order from the project folder:

```bash
# Task 1 — takes ~2 mins
python3 experiments.py

# Task 2 — takes ~4 mins  
python3 task2_experiments.py

# Task 3 — takes ~5 mins
python3 task3_experiments.py
```

Each script is fully self-contained and prints all results to the terminal.

---

# 📚 PDF Reading Guide

| Situation | PDF to Read |
|-----------|-------------|
| Quick professor presentation | `Complete_Project_All_Tasks.pdf` |
| Deep study for Task 1 | `Task1_Complete_Study_Guide.pdf` |
| Formal academic report (Task 1) | `Task1_ComHAI_PLACO_Report.pdf` |
| Fairness concepts + Task 2 results | `Task2_Fairness_Study_Guide.pdf` |
| Task 3 algorithm theory + Q&A | `Task3_Novel_Algorithms_Study_Guide.pdf` |

---

*Fairness in Human-AI Teams Project — IIT Ropar — 2025-26*