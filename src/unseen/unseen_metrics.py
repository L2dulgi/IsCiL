import numpy as np
import re
import argparse
import os
from clus.env.continual_config import *

DEFAULT_MAX_SCORE = 4.0

class ContinualMetricCalculator:
    def __init__(self, file_path='./data/test.pkl', mode='mmworld', detailed=False):
        self.file_path = file_path
        self.mode = mode
        self.detailed = detailed
        self.data = None
        self.phases = 0
        # Dictionary to store each task's info (FWT, BWT, AUC, etc.)
        self.metric_dict = {}

    def calculate_metrics(self, suffix=""):
        """
        Calculate BWT (Backward Transfer), FWT (Forward Transfer),
        and AUC (average performance after learning) in percentages.
        """
        # Gather final data per phase
        parsed_phases = []
        for phase_data in self.data:
            if isinstance(phase_data, list):
                # If multiple entries exist for a phase, take the last one
                parsed_phases.append(phase_data[-1])
            else:
                parsed_phases.append(phase_data)

        # Build self.metric_dict by analyzing each phase
        for phase_idx, phase_dict in enumerate(parsed_phases):
            train_tasks = phase_dict.get('train', [])
            eval_data = phase_dict.get('eval', {})

            for ttask in train_tasks:
                # Check if it appears in eval
                if ttask not in eval_data:
                    continue

                # If already in metric_dict, rename to avoid collision
                task_key = ttask
                if task_key in self.metric_dict:
                    task_key = f"{ttask}_{phase_idx}"

                # Initialize storage
                self.metric_dict[task_key] = {
                    'bwt_score': 0.0,      # Average difference to measure backward transfer
                    'learned_phase': phase_idx,
                    'fwt_score_init': 0.0, # The baseline (score upon learning)
                    'bwt_scores': [],      # List of (future_phase_idx, difference_from_learned)
                    'auc_score': 0.0,      # Mean of the normalized scores across phases
                    'auc_scores': [],      # List of (phase_idx, normalized_metric)
                }

                # The metric when the task was first learned
                learned_raw, learned_max = self._get_scores(eval_data, ttask)
                learned_norm = learned_raw / learned_max
                self.metric_dict[task_key]['fwt_score_init'] = learned_norm

                # For subsequent phases: measure how the normalized metric changes
                for f_phase_idx in range(phase_idx, self.phases):
                    future_eval_data = self.data[f_phase_idx].get('eval', {})
                    if ttask not in future_eval_data:
                        continue
                    future_raw, future_max = self._get_scores(future_eval_data, ttask)
                    future_norm = future_raw / future_max

                    # BWT difference after the original phase
                    if f_phase_idx != phase_idx:
                        diff = future_norm - learned_norm
                        self.metric_dict[task_key]['bwt_scores'].append((f_phase_idx, diff))
                    # Keep track of this normalized score for AUC
                    self.metric_dict[task_key]['auc_scores'].append((f_phase_idx, future_norm))

                # Convert to arrays for averaging
                bwt_arr = np.array(self.metric_dict[task_key]['bwt_scores'])
                auc_arr = np.array(self.metric_dict[task_key]['auc_scores'])

                if phase_idx == (len(parsed_phases) - 1):
                    # If it's the last phase of training for this task, we won't compute future BWT
                    if len(auc_arr) > 0:
                        # Just store the first AUC value
                        self.metric_dict[task_key]['auc_score'] = auc_arr[0, 1]
                else:
                    # Compute average BWT
                    if len(bwt_arr) > 0:
                        bwt_mean = np.mean(bwt_arr[:, 1])  # average difference
                        self.metric_dict[task_key]['bwt_score'] = bwt_mean

                    # Compute average AUC
                    if len(auc_arr) > 0:
                        auc_mean = np.mean(auc_arr[:, 1])
                        self.metric_dict[task_key]['auc_score'] = auc_mean

        # Compute global averages (in percentage)
        bwt_sum, bwt_cnt = 0.0, 0
        fwt_sum, fwt_cnt = 0.0, 0
        auc_sum, auc_cnt = 0.0, 0

        for task_name, vals in self.metric_dict.items():
            bwt_sum += vals['bwt_score']
            bwt_cnt += 1
            fwt_sum += vals['fwt_score_init']
            fwt_cnt += 1
            auc_sum += vals['auc_score']
            auc_cnt += 1

        # Convert to percentages
        bwt_mean = (bwt_sum / bwt_cnt * 100) if bwt_cnt else 0
        fwt_mean = (fwt_sum / fwt_cnt * 100) if fwt_cnt else 0
        auc_mean = (auc_sum / auc_cnt * 100) if auc_cnt else 0

        # Return results for storage
        return {
            'BWT': bwt_mean,
            'FWT': fwt_mean,
            'AUC': auc_mean,
            'suffix': suffix
        }

    def _get_scores(self, eval_data, ttask):
        """
        Returns (raw_score, max_score). If the stored value is just a float,
        we assume default max of 4. Otherwise, it's a tuple of (raw, max).
        """
        val = eval_data[ttask]
        if isinstance(val, tuple) and len(val) == 2:
            raw_score, max_score = val
            return raw_score, max_score
        else:
            # old style or no max
            return float(val), DEFAULT_MAX_SCORE

    def filter_tasks_by_mode(self, data, mode, env):
        """
        Filter tasks based on the mode and environment settings.
        """
        if env == 'mmworld':
            if mode == 'ind':
                print('[ind mode processing ... ]')
                for p in data:
                    p['train'] = [pl for pl in p['train'] if pl not in UNSEEN_UNSEEN]
            elif mode == 'indo':
                print('[indo mode processing ... ]')
                for p in data:
                    p['train'] = [pl for pl in p['train'] if pl in UNSEEN_IND]
            elif mode == 'unseen':
                print('[unseen mode processing ... ]')
                for p in data:
                    p['train'] = [pl for pl in p['train'] if pl not in UNSEEN_IND]
            elif mode == 'unseeno':
                print('[unseeno mode processing ... ]')
                for p in data:
                    p['train'] = [pl for pl in p['train'] if pl in UNSEEN_UNSEEN]
            elif mode == 'original':
                print('[original mode processing ... ]')
                for p in data:
                    p['train'] = [pl for pl in p['train'] if pl not in UNSEEN_IND and pl not in UNSEEN_UNSEEN]
        elif env == 'kitchen':
            if mode == 'indo':
                print('[kitchen indo mode processing ... ]')
                for p in data:
                    p['train'] = [pl for pl in p['train'] if pl in KITCHEN_UNSEEN]
        return data


def from_logging(logging_file_path):
    """
    Parses the log file into a list of dicts:
    [
        {
          'train': [taskA, taskB, ...],
          'eval': {
              'taskA': (score, max_score) or just float
              ...
          }
        },
        ...
    ]
    Handles both:
    - old style "[0]skill is ... rew : 3.67"   => stored as (3.67, 4.0) by default
    - new style "[task 0] sub_goal sequence is ... task GC : 66.67% (2.67 / 4.00)"
      => stored as (raw_score, max_score)
    """
    with open(logging_file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip().split(' ') for line in lines if line.strip()]

    phase_metric_list = []
    current_phase_dict = None

    # Regex for new evaluation lines
    # e.g. "[task 0] sub_goal sequence is ['microwave', 'kettle'] task GC : 66.67% (2.67 / 4.00)"
    pattern_new_eval = re.compile(
        r"\[task\s+(\d+)\].*sub_goal sequence is (.*)\s+task GC :\s+([\d\.]+)% \(([\d\.]+)\s*/\s*([\d\.]+)\)"
    )

    for line in lines:
        joined_line = " ".join(line)

        # Start of a new phase from "{'data_name': ...}"
        if line and line[0].startswith("{'data_name':"):
            if "'data_paths':" in line:
                dp_idx = line.index("'data_paths':")
                task_line = '-'.join(line[1:dp_idx]).replace("'", '')
            else:
                task_line = line[1].replace("'", '')

            task_line = task_line.strip(',').split(',')
            task_line = [item for item in task_line if item]

            current_phase_dict = {
                'train': task_line,
                'eval': {}
            }
            phase_metric_list.append(current_phase_dict)
            continue

        # If no phase has started, skip
        if not phase_metric_list:
            continue

        # OLD STYLE: "[0]skill is ... rew : X"
        if 'skill' in line[0] and len(line) > 2 and line[1] == 'is':
            try:
                metric_val = float(line[-1])
            except ValueError:
                continue

            # The skill names are in line[3:-2], e.g. ['microwave', 'kettle']
            skill_chunk = line[3:-2]
            skill_chunk = [item.strip(",'[]") for item in skill_chunk if item]
            joined_task = '-'.join(skill_chunk)
            # Store as (raw_score, default_max_score)
            current_phase_dict['eval'][joined_task] = (metric_val, DEFAULT_MAX_SCORE)

        # NEW STYLE: "[task 0] ... task GC : 66.67% (2.67 / 4.00)"
        elif '[task' in line[0] and 'sub_goal' in joined_line and 'task GC' in joined_line:
            match = pattern_new_eval.search(joined_line)
            if match:
                subgoals = match.group(2).strip()
                raw_score_str = match.group(4)
                max_score_str = match.group(5)

                try:
                    raw_score = float(raw_score_str)
                    max_score = float(max_score_str)
                except ValueError:
                    raw_score, max_score = 0.0, DEFAULT_MAX_SCORE

                # Convert subgoals to a single task name
                clean_subgoals = subgoals.strip("[] ").replace("'", "")
                clean_subgoals = clean_subgoals.replace(",", "").replace("  ", " ")
                joined_task = "-".join(clean_subgoals.split())

                # Store as (raw_score, max_score)
                current_phase_dict['eval'][joined_task] = (raw_score, max_score)

    # Remove phases with no 'eval' data
    final_phases = []
    for p_dict in phase_metric_list:
        if 'eval' in p_dict and len(p_dict['eval']) > 0:
            final_phases.append(p_dict)

    return final_phases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Continual Learning metric calculation for unseen tasks.'
    )
    parser.add_argument('-al', '--algo', type=str, default='iscil',
                        help='Algorithm name')
    parser.add_argument('-e', '--env', type=str, default='kitchen',
                        help='Environment (mmworld/kitchen)')
    parser.add_argument('-p', '--path', type=str, default='iscil',
                        help='Path type')
    parser.add_argument('-g', '--grep', type=str, default='',
                        help='Filter directories containing this string')
    parser.add_argument('-m', '--mode', type=str, default=None,
                        help='Unseen evaluation mode (ind/indo/unseen/unseeno/original)')
    parser.add_argument('--detailed', action='store_true',
                        help='If set, print per-task BWT/FWT/AUC in detail.')

    args = parser.parse_args()

    algo = args.algo
    env = args.env

    # Path setup
    etype = 'Unseen_experiments'
    base_path = f'data/{etype}/{algo}/{env}/ius'

    if not os.path.isdir(base_path):
        print(f"[Error] Base path not found: {base_path}")
        exit(1)

    # Find evaluation paths
    eval_dirs = [d for d in os.listdir(base_path) if args.grep in d]
    eval_paths = [os.path.join(base_path, d, 'training_log.log') for d in eval_dirs]
    eval_paths.sort()

    print(f"Found {len(eval_paths)} log files to process")

    # Create calculator instance
    cal = ContinualMetricCalculator(mode=args.mode, detailed=args.detailed)

    # Process each log file
    for log_path in eval_paths:
        if not os.path.isfile(log_path):
            continue

        print(f"\nProcessing log file: {log_path}")
        try:
            # Parse log data once
            log_data = from_logging(log_path)
            
            # Store results
            results = []
            
            # Calculate normal metrics (no filtering or original mode)
            import copy
            normal_data = copy.deepcopy(log_data)
            if args.mode == 'original' and env == 'mmworld':
                normal_data = cal.filter_tasks_by_mode(normal_data, 'original', env)
            cal.data = normal_data
            cal.phases = len(normal_data)
            cal.metric_dict = {}  # Reset metric dict
            normal_results = cal.calculate_metrics()
            results.append(normal_results)
            
            # Calculate indo metrics
            indo_data = copy.deepcopy(log_data)
            indo_data = cal.filter_tasks_by_mode(indo_data, 'indo', env)
            cal.data = indo_data
            cal.phases = len(indo_data)
            cal.metric_dict = {}  # Reset metric dict
            indo_results = cal.calculate_metrics("-A")
            results.append(indo_results)
            
            # Print combined results
            print("========================================")
            print(" Continual Learning Metrics (in %) ")
            print("========================================")
            
            # Print normal metrics
            print(f"BWT (Backward Transfer): {results[0]['BWT']:.2f}%")
            print(f"FWT (Forward Transfer) : {results[0]['FWT']:.2f}%")
            print(f"AUC (Average Score)    : {results[0]['AUC']:.2f}%")
            print("----------------------------------------")
            
            # Print indo metrics with -A suffix
            print(f"BWT-A (Unseen Only)      : {results[1]['BWT']:.2f}%")
            print(f"FWT-A (Unseen Only)      : {results[1]['FWT']:.2f}%")
            print(f"AUC-A (Unseen Only)      : {results[1]['AUC']:.2f}%")
            print("========================================")
            
        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            import traceback
            traceback.print_exc()