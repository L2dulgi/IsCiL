from inspect import isfunction
import jax

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def update_rngs(rngs) :
    '''
    update rngs for next step
    :params rngs : dict of rngs
    '''
    for keys in rngs.keys() :
        rngs[keys] , _ = jax.random.split(rngs[keys])
    return rngs

