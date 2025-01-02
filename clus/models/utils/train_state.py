# from transformers.models.gpt2.modeling_flax_gpt2 import * 
from flax.training import train_state

import jax.random as random
import optax

from clus.models.base.conditional_block import *

import optax

def create_train_state_time_cond( model, input_config, optimizer_config):
    # model rng keys initailization 
    # k1, k2 = random.split(random.PRNGKey(444), 2)
    # r1, r2 = random.split(random.PRNGKey(777), 2)
    rngs = {'params': random.PRNGKey(444), 'dropout':random.PRNGKey(44)}

    input_kwargs = {}
    for k in input_config.keys():
        input_kwargs[k] = jnp.zeros(input_config[k])

    params = model.init(rngs, **input_kwargs)

    # # optimizer initializer magic parameters ( 1024 base training )
    optimizer_cls = optimizer_config['optimizer_cls']
    optimizer = optimizer_cls(**optimizer_config['optimizer_kwargs'])

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def create_train_state_basic(model, input_config, optimizer_config=None):
    # model rng keys initailization 
    k1, k2 = random.split(random.PRNGKey(444), 2)
    r1, r2 = random.split(random.PRNGKey(777), 2)
    rngs = {'params': k1, 'dropout':r1}
    
    input_dict = {}
    for key in input_config.keys():
        input_dict[key] = jnp.zeros(input_config[key])

    params = model.init(rngs=rngs, **input_dict)

    # optimizer initializer
    if optimizer_config is None :
        lr = 1e-5
        momentum = 0.9
        optimizer = optax.adam(lr, momentum)
    else :
        optimizer_cls = optimizer_config['optimizer_cls']
        optimizer = optimizer_cls(**optimizer_config['optimizer_kwargs'])

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)