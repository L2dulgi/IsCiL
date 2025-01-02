

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
import optax

def create_default_mask(params):
    def _map(params, mask):
        for k in params:
            if isinstance(params[k], FrozenDict):
                mask[k] = {}
                _map(params[k], mask[k])
            else:
                mask[k] = 'zero'
    mask = {}
    _map(params, mask)
    return frozen_dict.freeze(mask)

def create_finetuning_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'adam'
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'zero'
    mask = {}
    _map(params, mask, label_fn)
    return frozen_dict.freeze(mask)

def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'zero'
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'adam'
    mask = {}
    _map(params, mask, label_fn)
    return frozen_dict.freeze(mask)

# freeze layer from pretrained model
def zero_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

## first_depth is 0
def compare_params(lhs, rhs, depth=0):
    for k in lhs.keys():
        if isinstance(lhs[k], FrozenDict):
            print('  ' * depth, k)
            compare_params(lhs[k], rhs[k], depth + 1)
        else:
            print('  ' * depth, k, jnp.mean(jnp.abs(lhs[k] - rhs[k])))


