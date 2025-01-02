
import os 
def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")
        if 'test' in path:
            print("Exiting. but test is in path")
        else : 
            exit()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


import jax.numpy as jnp
import json
# from clus.models.peftpool.lorax_utils import *

def print_params_shapes(params):
    def extract_shapes(params):
        out = {}
        for key, value in params.items():
            if isinstance(value, dict):
                out[key] = extract_shapes(value)
            elif isinstance(value, jnp.ndarray):
                out[key] = str(value.shape)
            
            # elif isinstance(value, LoraWeightPool) :
            #     out[key] = str(value.pool_mask)
        return out

    formatted_output = json.dumps(extract_shapes(params), indent=2)
    print(formatted_output)

import json
from jax import numpy as jnp

def compare_param_values(params1, params2):
    def compare_values(p1, p2, key_path=''):
        differences = {}
        # Ensure we are comparing the same keys in both parameter sets
        all_keys = set(p1.keys()).union(p2.keys())
        for key in all_keys:
            full_key = f'{key_path}.{key}' if key_path else key
            if key not in p1:
                differences[full_key] = 'missing in params1'
            elif key not in p2:
                differences[full_key] = 'missing in params2'
            else:
                val1, val2 = p1[key], p2[key]
                if isinstance(val1, dict) and isinstance(val2, dict):
                    # Recursive comparison for nested dictionaries
                    sub_differences = compare_values(val1, val2, full_key)
                    if sub_differences:
                        differences.update(sub_differences)
                elif isinstance(val1, jnp.ndarray) and isinstance(val2, jnp.ndarray):
                    # Direct comparison for jnp.ndarray objects
                    if not jnp.array_equal(val1, val2):
                        differences[full_key] = 'arrays differ'
                else:
                    # Fallback for comparing non-dictionary, non-array objects
                    pass
                    # if val1 != val2:
                    #     differences[full_key] = 'values differ'
                
        return differences

    # Run the comparison
    value_differences = compare_values(params1, params2)
    print(value_differences )

    # Output the result
    if value_differences:
        print("Differences found in parameter values:")
        for diff_key, diff_val in value_differences.items():
            print(f"{diff_key}: {diff_val}")
    else:
        print("No differences in parameter values.")

