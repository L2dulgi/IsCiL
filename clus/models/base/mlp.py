# import torch
import numpy as np
import transformers
from transformers.models.gpt2.modeling_flax_gpt2 import * 

import wandb
# from VLEmbedder import videoCLIPExtractor
from flax.training import train_state
import jax.random as random
import optax
import os
import random as py_rand

# from train_state import VQTrainState
from einops import rearrange, repeat
# from clip_embedding import HARD_CODED_CLIP_EMBED 

class MLPdecoder(nn.Module):
    hidden_size: int=256
    out_shape: int=4
    dropout_rate : float=0.1
    deterministic: bool=False

    def setup(self) -> None:
        self.layer0 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
        ])

        self.layer1 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
        ])

        self.layer2 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
        ])

        self.layer3 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
        ])

        self.layer_out = nn.Sequential([
            nn.Dense(self.out_shape),
        ])


        self.t_mlp = nn.Sequential([
            nn.Dense(32),
            nn.gelu,
            nn.Dense(32),
        ])

    def __call__(self, x , t, cond):
        t_emb = self.t_mlp(t)
        x = jnp.concatenate((x, cond, t_emb), axis=-1)
        x = self.layer0(x)
        res = x 
        x = self.layer1(x) + res
        res = x
        x = self.layer2(x) + res
        res = x
        x = self.layer3(x) + res
        x = self.layer_out(x)
        return x

class CondMLP(nn.Module):
    hidden_size: int=256
    time_shape : int=32
    out_shape: int=4
    dropout_rate : float=0.1
    deterministic: bool=False

    def setup(self) -> None:
        
        self.hidden_embed = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
        ])

        self.layer0 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.out_shape),
        ])


        self.t_mlp = nn.Sequential([
            nn.Dense(32),
            nn.gelu,
            nn.Dense(32),
        ])

    def __call__(self, x , t, cond, deterministic=False):
        # t_emb = self.t_mlp(t)
        x = jnp.concatenate((x, cond, t), axis=-1)
        x = self.layer0(x)
        return x


class MLP(nn.Module):
    '''
    tanh is included
    '''
    hidden_size: int=256
    out_shape: int=4
    dropout_rate : float=0.1
    deterministic: bool=False

    def setup(self) -> None:
        self.layer0 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            # nn.relu,
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.out_shape),
            nn.tanh,
        ])

    def __call__(self, x):
        x = self.layer0(x)
        return x
    

class NormalMLP(nn.Module):
    hidden_size: int=256
    out_shape: int=4
    dropout_rate : float=0.1
    deterministic: bool=False

    def setup(self) -> None:
        self.layer0 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            # nn.relu,
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.out_shape),
        ])

    def __call__(self, x):
        x = self.layer0(x)
        return x
    

class SinuosidualMLP(nn.Module) :
    hidden_size: int=256
    out_shape: int=4
    dropout_rate : float=0.1
    deterministic: bool=False

    def setup(self) -> None:
        self.layer0 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.gelu,
            nn.Dense(self.out_shape),
        ])

    def __call__(self, x):
        x = sinusodial_embedding_flat(x, 32)
        x = self.layer0(x)
        return x
    

def sinusodial_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=np.float32) / half
        )
        args = timesteps[:, jnp.newaxis] * freqs[jnp.newaxis]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

def sinusodial_embedding_flat(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=np.float32) / half
        )
        args = timesteps[...,jnp.newaxis] * freqs[jnp.newaxis]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)

    embedding = rearrange(embedding, 'b f u -> b (f u)')
    return embedding