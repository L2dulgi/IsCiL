# Incremental Learning of Retrievable Skills for Efficient Continual Task Adaptation (IsCiL)

This repository provides the official implementation of our paper:
[Incremental Learning of Retrievable Skills for Efficient Continual Task Adaptation](https://openreview.net/pdf?id=RcPAJAnpnm)

Poster : [NeurIPS2024](https://neurips.cc/virtual/2024/poster/95159)

#Incremental Learning #Imitation Learning #Skills #NeurIPS2024

---

## Overview

![](fig/fig_concept.png)

**Continual Imitation Learning (CiL)** involves extracting and accumulating task knowledge from demonstrations across multiple stages and tasks to achieve a multi-task policy. With recent advancements in foundation models, there has been a growing interest in **adapter-based CiL approaches**, where adapters are introduced in a parameter-efficient way for newly demonstrated tasks. While these approaches effectively isolate parameters for different tasks—helping mitigate catastrophic forgetting—they often limit **knowledge sharing** across tasks.

We introduce **IsCiL**, an **adapter-based CiL framework** that addresses the limitation of knowledge sharing by incrementally learning **shareable skills** from different demonstrations. This enables **sample-efficient task adaptation**, especially in non-stationary CiL environments. In IsCiL, demonstrations are mapped into a state embedding space, where **proper skills** can be retrieved from a **prototype-based memory**. These retrievable skills are then incrementally refined on their own **skill-specific adapters**. Our experiments on complex tasks in **Franka-Kitchen** and **MetaWorld** demonstrate robust performance of IsCiL in both **task adaptation** and **sample efficiency**. Additionally, we provide a simple extension of IsCiL for **task unlearning** scenarios.

![](fig/fig_method.png)

**Implementation highlights:**
- **Incremental creation of skill-specific adapters.**
  1. **K-means** is used to build skill bases, improving the accuracy of similarity searches between inputs and the corresponding skill.
  2. **Evaluation** is performed by applying each skill adapter to a pre-trained model, enabling effective handling of new or changing inputs.
---

## Table of Contents

1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Dataset and Environment Setup](#dataset-and-environment-setup)
4. [Running the Experiments](#running-the-experiments)
5. [Baselines and Implementation Details](#baselines-and-implementation-details)
6. [Evaluation](#evaluation)
7. [Error Management and Troubleshooting](#error-management-and-troubleshooting)

---

## Installation

### Note
`clus` is the development version name for `iscil`. All related implementations are located inside the `clus` directory.

### Requirements
- Python 3.10.13
- [mujoco210](https://github.com/openai/mujoco-py)

### Step 1: Create a Conda Environment

1. Create and activate a conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate iscil
   ```
2. Verify successful activation:
   ```bash
   conda info --envs
   ```

### Step 2: Install Specific Package Versions
We rely on a specific version of `gym`:
```bash
pip install setuptools==65.5.0 "wheel<0.40.0"    # Prevents an error when installing gym 0.21.0
pip install gym==0.21.0
```
For more details on why the first line is required, see:  
[Why is pip install gym failing?](https://stackoverflow.com/questions/76129688/why-is-pip-install-gym-failing-with-python-setup-py-egg-info-did-not-run-succ)

### Step 3: Install the Project
In your project directory, run:
```bash
pip install -e .
```

### Step 4: Set Environment Variables
Make sure the following environment variables are set:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

## Environment Setup

If the steps above are completed, you can test the Kitchen environment directly. For Metaworld, follow the instructions below.

### Download and Organize Datasets and Environments

1. Download environment files from the following link:  
   [Google Drive](https://drive.google.com/file/d/1x9FjohGHyultblhQFZXXLs_FhaTYlg5F/view?usp=drive_link)

2. After downloading:
   - Unzip the file and locate the `IsCiL_Env/data` folder.
   - For each environment folder, install it:
     ```bash
     # For mmworld
     cd IsCiL_Env/env/mmworld
     pip install -e .

     # For Metaworld
     cd ../Metaworld
     pip install -e .
     ```
   - Return to the root directory if necessary.

---

## Dataset and Environment Setup

The above steps cover both dataset and environment requirements (e.g., Kitchen, Metaworld). Make sure you have everything installed before proceeding.

---

## Running the Experiments

### Download Pre-Trained Models
1. Download the `pre_trained_models.zip` file from the provided link:  
   [Download](https://drive.google.com/file/d/1KbYd3hJWG6yr0sKuB9i4WmVDCS5Uefz1/view)
2. Move the file to `data` and unzip it.
3. Check that the contents are properly placed in the `data/pre_trained_models` directory.

### Run IsCiL
To run the IsCiL experiment, navigate to the `clus` directory and execute:
```bash
bash src/IsCiL.sh
```
This script will launch the incremental learning process described in the paper.


---

## Evaluation

IsCiL provides comprehensive evaluation capabilities to assess both standard task performance and generalization to unseen tasks. The evaluation framework measures continual learning metrics including Backward Transfer (BWT), Forward Transfer (FWT), and Area Under the Curve (AUC).

### [Optional] Standard Evaluation (already included on training)

To evaluate trained models on tasks seen during training to create the log files:

```bash
python src/l2m_evaluator.py --save_id <experiment_id>
```

This will:
- Load saved models from each training stage
- Evaluate performance on all learned tasks
- Generate evaluation logs and metrics files

**Example:**
```bash
python src/l2m_evaluator.py --save_id HelloIsCiL_complete_0
```

### Metrics 
Calculate continual learning metrics from evaluation logs:
```bash
python clus/utils/metric.py -al <algo> -g <grep string>
```
The `-g` parameter is used to filter the logs based on the grep string, which can be the experiment ID or any other identifier.

```
========================================
 Continual Learning Metrics (in %) 
========================================
BWT (Backward Transfer): XX.XX%
FWT (Forward Transfer) : XX.XX%
AUC (Average Score)    : XX.XX%
========================================
```


## Unseen Task Evaluation

To test generalization on unseen tasks, use the pretrained models and evaluate them on datastreams with unseen tasks:

```bash
python src/unseen/unseen_evaluator.py -e <env> -al <algo> -u <unseen_type> -id <evaluation_id>
```

**Parameters:**
- `-e/--env`: Environment (`kitchen` or `mmworld`)
- `-al/--algo`: Algorithm (`iscil`, `seq`, `ewc`, `mtseq`)
- `-id/--id`: Evaluation ID

**Example:**
```bash
python src/unseen/unseen_evaluator.py -e kitchen -al iscil -id HelloIsCiL_complete_0
```

### Metrics Calculation

Calculate continual learning metrics from evaluation logs:

```bash
python src/unseen/unseen_metrics.py -al <algo> 
```
The metrics calculator displays both overall metrics and unseen-only metrics (suffixed with -A):

```
========================================
 Continual Learning Metrics (in %) 
========================================
BWT (Backward Transfer): XX.XX%
FWT (Forward Transfer) : XX.XX%
AUC (Average Score)    : XX.XX%
----------------------------------------
BWT-A (Unseen Only)    : XX.XX%
FWT-A (Unseen Only)    : XX.XX%
AUC-A (Unseen Only)    : XX.XX%
========================================
```

### Interpreting Results

**Metrics Explanation:**
- **BWT (Backward Transfer)**: Measures knowledge retention. Negative values indicate forgetting, positive values indicate improvement
- **FWT (Forward Transfer)**: Initial performance on new tasks. Higher values indicate better knowledge transfer
- **AUC (Average Score)**: Overall performance across all tasks and phases

**Output Locations:**
- Training logs: `data/IsCiL_exp/<algo>/<env>/<id>/training_log.log`
- Unseen evaluation logs: `data/Unseen_experiments/<algo>/<env>/<unseen_type>/<id>/training_log.log`
- Model checkpoints: `data/IsCiL_exp/<algo>/<env>/<id>/models/`

---

## Baselines and Implementation Details

This section provides details about the baseline algorithms and implementation configurations used in our experiments.

### Available Algorithms

IsCiL is compared against several continual learning baselines. All algorithms can be specified using the `-al` parameter:

| Algorithm | Code | Description | Key Features |
|-----------|------|-------------|--------------|
| **IsCiL** | `iscil` | Our method | Incremental skill learning with dynamic LoRA adapters and multifaceted prototype retrieval |
| **Sequential LoRA** | `seqlora` | Sequential adapter learning | Single large LoRA adapter (dim=64) updated sequentially |
| **TAIL** | `tail` | Task-Adaptive Incremental Learning | Fixed small adapters (dim=16) with task-specific allocation |
| **TAIL-G** | `tailg` | TAIL with sub-goal id | sub-goal specific adapters (dim=4) |
| **L2M** | `l2m` / `l2mg` | Learn-to-Modulate | 100 learnable keys with small adapters (dim=4) |

### Implementation Details

#### IsCiL Configuration
- **Memory Pool**: 100 skill-specific adapters
- **LoRA Dimension**: 4 (parameter efficient)
- **Retrieval**: multiple bases for multifaceted skill retrieval
- **Key Features**:
  - Multifaceted-prototype generation based on K-means clustering
  - Meta-initialization from existing skills

#### Baseline Configurations

**Sequential LoRA (`seqlora`)**:
- Single LoRA adapter with dimension 64
- Updated continuously across all tasks
- No explicit skill separation

**TAIL (`tail`)**:
- Fixed allocation of adapters per task
- LoRA dimension: 16
- Task-specific adapter selection
- [Tricks] The implementation is the same as seq LoRA, and in the paper, the FWT calculated using metric.py is used as the AUC, since there is no forgetting(BWT=0).

**TAIL-G (`tailg`)**:
- Similar to TAIL but uses sub-goal level adapter selection

**L2M (`l2m`/`l2mg`)**:
- 100 learnable prototype keys
- Small LoRA adapters (dim=4)
- Similarity-based retrieval
- Variants: `l2m` (base embeddings), `l2mg` (split embeddings)

### Running Baseline Experiments

To run experiments with different algorithms:

```bash
# IsCiL (our method)
bash src/IsCiL.sh

# Run specific baseline
python src/train.py --algo seqlora --env kitchen --save_id baseline_seq_0

# Run baseline scripts
bash src/baselines.sh  # Runs all baseline comparisons
```
---

## Error Management and Troubleshooting

Occasionally, you might encounter errors when running the scripts. Below are common issues and how to fix them.

### Manual Library Dependency Troubleshooting

If errors occur while running:
```bash
bash src/IsCiL.sh
```
you can identify problematic imports by checking the console logs. Please note these possible fixes:

#### 1) Modify Library Imports (qax)
Some code modifications may be necessary due to updated JAX libraries:
- Replace occurrences of `jax.linear_util` with `jax.extend.linear_util` in:
  - `[qax package directory]/qax/implicit/implicit_array.py`
  - `[qax package directory]/qax/implicit/implicit_utils.py`

#### 2) D4RL Issues
If you get errors related to `abc`, ensure `collections.abc.Mapping` is used instead of `collections.Mapping` in:
  - `[D4RL package directory]/kitchen/adept_envs/mujoco_env.py`

#### 3) W&B Installation Error
If you encounter wandb installation errors, simply reinstall wandb:
```bash
pip install wandb
```

---

Enjoy exploring **Incremental Learning of Retrievable Skills for Efficient Continual Task Adaptation**!