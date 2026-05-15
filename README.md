# AI211: Machine Learning - Research Project
## IIT Ropar

This repository contains the research project work for the course **AI211: Machine Learning** at the **Indian Institute of Technology Ropar**.

### 👨‍🏫 Mentorship
*   **Mentor:** Shwetha Jain

### 📝 Project Overview
The core objective and details of this research work are outlined in the [Problem_Statement.pdf](./Problem_Statement.pdf) file. This project is a part of the academic curriculum for the 4th Semester.

For detailed execution, setup guidelines, and a step-by-step breakdown of each experiment, please refer to the **[Instructions.md](./Instructions.md)**.

### 📂 Repository Structure
*   **[Task_1/](./Task_1/):** Implementation of Complementary Human-AI (ComHAI) systems and the PLACO algorithm.
    *   `core.py`: All ComHAI + PLACO algorithm implementations.
    *   `simulate.py`: Simulates human annotation data.
    *   `experiments.py`: Runs all Task 1 reproducibility and extension experiments.
*   **[Task_2/](./Task_2/):** Fairness evaluation of Human-AI teams under various bias conditions.
    *   `fairness_metrics.py`: Implementation of 5 key fairness metrics (DP, EO, TPR gap, etc.).
    *   `biased_humans.py`: Simulation engine for 4 levels of human bias.
    *   `comhai_fair.py`: Binary ComHAI + PLACO adaptation.
    *   `task2_experiments.py`: Experiments studying fairness degradation as human bias increases.
*   **[Task_3/](./Task_3/):** Development of novel fairness-aware algorithms.
    *   `task3_algorithms.py`: Implementation of 5 novel fair algorithms (Fair-PLACO, Bias-Aware ComHAI, etc.).
    *   `task3_experiments.py`: Performance evaluation and lambda sensitivity analysis.
*   **[Problem_Statement.pdf](./Problem_Statement.pdf):** The foundational research problem description.
*   **[Instructions.md](./Instructions.md):** Detailed guide on running experiments for all three tasks.

### 🚀 Getting Started
1. Clone the repository.
2. Install requirements:
   ```bash
   pip install numpy scipy scikit-learn matplotlib seaborn pandas
   ```
3. Follow the specific instructions for each task:
    *   **Task 1:** `python Task_1/experiments.py`
    *   **Task 2:** `python Task_2/task2_experiments.py`
    *   **Task 3:** `python Task_3/task3_experiments.py`

### 📄 Documentation
Comprehensive study guides and technical reports are available for each task:
- **Master Document:** [Complete_Project_All_Tasks.pdf](./Complete_Project_All_Tasks.pdf)
- **Task 1:** [Task1_Report.pdf](./Task_1/Task1_Report.pdf), [Task1_Complete_Study_Guide.pdf](./Task_1/Task1_Complete_Study_Guide.pdf)
- **Task 2:** [Task2_Fairness_Study_Guide.pdf](./Task_2/Task2_Fairness_Study_Guide.pdf)
- **Task 3:** [Task3_Novel_Algorithms_Study_Guide.pdf](./Task_3/Task3_Novel_Algorithms_Study_Guide.pdf)

---
*Disclaimer: This project is strictly for academic and research purposes as part of the AI211 course at IIT Ropar.*
