## Descriptions

- **`/configs`**  
  Stores configuration files that set parameters and variables for the project.

- **`/env`**  
  Houses environment setup files, such as scripts for evaluation environments or container configurations.

- **`/models`**  
  Includes all model-related files, potentially sorted into different categories or types:
  - **`/base`**  
    Model templates or classes that serve as a foundation for further model development.
  - **`/model`**  
    The main models that implement training and evaluation.
  - **`/peftpool`**  
    Contains most of the IsCiL implementation, including the skill retriever and adapters.

- **`/trainer`**  
  Contains code for Continual Imitation Learning (CiL) training scenarios.

- **`/utils`**  
  Provides code for computing evaluation metrics and basic loss functions for model training.



# Continual Imitation Learning Metric Calculator (utils/metric.py)

This directory contains a Python script that parses log files produced during continual imitation learning experiments and computes several metrics:

- **BWT (Backward Transfer)**  
- **FWT (Forward Transfer)**  
- **AUC (Average Score)**  

All metrics are calculated as percentages, and logs can contain both old-style and new-style entries.

---

## Contents

1. [Introduction](#introduction)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Explanation of Metrics](#explanation-of-metrics)  
5. [Example](#example)

---

## Introduction

When doing **Continual Imitation Learning (CiL)**, it is often useful to evaluate:

- **How well newly learned tasks maintain performance on previously learned tasks** (BWT).  
- **How quickly new tasks are learned from the outset** (FWT).  
- **Overall performance across all tasks** (AUC).

This script reads a `training_log.log` file containing either or both:

1. **Old-Style Logs**:  
   ```
   [0]skill is  ['microwave', 'kettle'] rew : 3.67
   ```
   - Interpreted as a raw score of `3.67` out of `4.0` (the default maximum).

2. **New-Style Logs**:  
   ```
   [task 0] sub_goal sequence is ['microwave', 'kettle'] task GC : 66.67% (2.67 / 4.00)
   ```
   - Extracts a raw score of `2.67` out of a max score of `4.00`.  

Both formats can appear in the same file, and the script **automatically** handles them.

---

## Usage

In a terminal, run:
```bash
python your_metric_script.py \
    -al <ALGORITHM_NAME> \
    -e <ENVIRONMENT_NAME> \
    [-p <PATH_OPTION>] \
    [-g <GREP_PATTERN>] \
    [--detailed]
```

**Command-Line Arguments:**

- `-al, --algo <str>`  
  Name or identifier of the algorithm (default: `cilu`).

- `-e, --env <str>`  
  Name of the environment (default: `kitchen`).

- `-p, --path <str>`  
  Not strictly required in this script, but if your code uses it to find a different subfolder, you can adjust it (default: `cilu`).

- `-g, --grep <str>`  
  When reading from the base directory, only directories containing this substring in their name will be considered.

- `--detailed`  
  If set, prints the per-task metrics in addition to the global summary metrics.

**Key Points:**

1. The script expects a directory structure like:
   ```
   data/IsCiL_exp/<algo>/<env>/
   ├─ experiment_a/
   │   └─ training_log.log
   ├─ experiment_b/
   │   └─ training_log.log
   └─ ...
   ```
2. For each directory under `.../<algo>/<env>/` that matches the grep pattern, the script loads `training_log.log` and processes it.
3. If a log line has an old-style format, e.g.:
   ```
   [0]skill is  ['microwave', 'kettle'] rew : 3.67
   ```
   it’s treated as `(3.67 / 4.0)` by default.
4. If a log line has the new-style format, e.g.:
   ```
   [task 0] sub_goal sequence is ['microwave', 'kettle'] task GC : 66.67% (2.67 / 4.00)
   ```
   it’s treated as `(2.67 / 4.00)`.
5. The script then computes **Backward Transfer (BWT)**, **Forward Transfer (FWT)**, and an **AUC**-like metric for each task and across all tasks.

---

## Explanation of Metrics

- **Forward Transfer (FWT)**:  
  The initial performance on a newly learned task, normalized by its maximum possible score (e.g., `raw_score / max_score`).  
  - Reported in `%`.

- **Backward Transfer (BWT)**:  
  The average change in performance (difference from learned-phase performance) on that task across future phases.  
  - If you lose performance on previously learned tasks, BWT is negative; if you gain, it’s positive.  
  - Reported in `%`.

- **AUC (Average Score)**:  
  The mean of a task’s normalized scores across all phases after it was learned.  
  - Also reported in `%`.  
  - Higher indicates the method maintains (or improves) performance over time.

The script aggregates these across all tasks to give an overall **BWT**, **FWT**, and **AUC**. If `--detailed` is provided, each task’s metrics are also printed.

---

## Example

Assuming you have your logs in:
```
data/IsCiL_exp/cilu/kitchen/exp1/training_log.log
data/IsCiL_exp/cilu/kitchen/exp2/training_log.log
```
You could run:
```bash
python your_metric_script.py \
    -al cilu \
    -e kitchen \
    -g exp \
    --detailed
```
This would:
1. Find `exp1/` and `exp2/` under `data/IsCiL_exp/cilu/kitchen/`.
2. Parse their `training_log.log` files.
3. Print overall and per-task (because of `--detailed`) BWT, FWT, AUC.

Sample output:
```
Processing log file: data/IsCiL_exp/cilu/kitchen/exp1/training_log.log
========================================
 Continual Learning Metrics (in %) 
========================================
BWT (Backward Transfer): 5.30%
FWT (Forward Transfer) : 88.25%
AUC (Average Score)    : 85.40%
========================================

--- Per-Task Detailed Metrics ---
Task: microwave-kettle
  BWT: 3.75%
  FWT: 90.00%
  AUC: 88.12%
--------------------------------
Task: microwave-bottom burner
  BWT: -1.20%
  FWT: 86.50%
  AUC: 85.00%
--------------------------------

Processing log file: data/IsCiL_exp/cilu/kitchen/exp2/training_log.log
...
```

---

Enjoy monitoring **BWT**, **FWT**, and **AUC** with these logs to track the performance of your **Continual Imitation Learning** setups!