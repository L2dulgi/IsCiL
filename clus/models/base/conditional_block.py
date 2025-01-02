import numpy as np
from transformers.models.gpt2.modeling_flax_gpt2 import * 

import wandb

from flax.training import train_state
import jax.random as random
import optax
import os
import random as py_rand
from einops import rearrange, repeat

#################### transformer block ####################
class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return hidden_linear * nn.gelu(hidden_gelu)

class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of Pynp's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states

class FlaxCrossAttention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        # compute attentions
        attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
        attention_scores = attention_scores * self.scale
        attention_probs = nn.softmax(attention_scores, axis=2)

        # attend to values
        hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        return hidden_states

class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        # cross attention
        self.attn2 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states

class FlaxTimeCondTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    out_dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.t_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.hidden_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.context_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.out = nn.Dense(self.out_dim, dtype=self.dtype)

        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        
        # cross attention
        self.attn2 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        self.attn3 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)

        # time conditional attention
        self.ff1 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.ff2 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.ff3 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)

        self.norm_at1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_at2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_at3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        self.norm_ff1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_ff2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_ff3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, time, context, deterministic=False):
        # self attention
        t_emb = self.t_emb(time)
        hidden_states = self.hidden_emb(hidden_states) + t_emb
        context = self.context_emb(context)
        
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm_at1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm_at1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff1(self.norm_ff1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm_at2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward 2
        residual = hidden_states
        hidden_states = self.ff3(self.norm_ff3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention 2
        residual = hidden_states
        hidden_states = self.attn3(self.norm_at3(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward last
        residual = hidden_states
        hidden_states = self.ff2(self.norm_ff2(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        hidden_states = self.out(hidden_states)

        return hidden_states

class FlaxMultiCondTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    out_dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.t_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.hidden_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.context_emb_1 = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.context_emb_2 = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.out = nn.Dense(self.out_dim, dtype=self.dtype)

        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        
        # cross attention
        self.attn2 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        self.attn3 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)

        # time conditional attention
        self.ff1 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.ff2 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.ff3 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)

        self.norm_at1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_at2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_at3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        self.norm_ff1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_ff2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_ff3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, time, context_1, context_2, deterministic=False):
        # self attention
        t_emb = self.t_emb(time)
        hidden_states = self.hidden_emb(hidden_states) + t_emb
        context_1 = self.context_emb_1(context_1)
        context_2 = self.context_emb_2(context_2)
        context = jnp.concatenate([context_1, context_2], axis=1) # concat context_1 and context_2 (B,S,F) in sequence
        
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm_at1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm_at1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff1(self.norm_ff1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm_at2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward 2
        residual = hidden_states
        hidden_states = self.ff3(self.norm_ff3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention 2
        residual = hidden_states
        hidden_states = self.attn3(self.norm_at3(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward last
        residual = hidden_states
        hidden_states = self.ff2(self.norm_ff2(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        hidden_states = self.out(hidden_states)

        return hidden_states

#################### vector as seq ####################
from einops import rearrange, repeat

#################### timesteps ####################
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
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

def sinusodial_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only: # TODO check values
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=np.float32) / half
        )
        args = timesteps[:] * freqs[jnp.newaxis]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)

        print(embedding[0,0])
    return embedding

#################### Refactored Transformers ####################

class FlaxDenoisingTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer 
    """

    # internal dimension for transformer block
    dim: int
    out_dim: int
    n_heads: int
    d_head: int
    context_emb_dim: int=512
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.t_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.hidden_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.context_emb = nn.Sequential([
            nn.Dense(self.context_emb_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.out = nn.Dense(self.out_dim, dtype=self.dtype)

        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        self.attn2 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)
        # self.attn3 = FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype)

        self.norm_at1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_at2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        # self.norm_at3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        self.ff1 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.ff2 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        # self.ff3 = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)

        self.norm_ff1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm_ff2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        # self.norm_ff3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, time, context, deterministic=False):
        # self attention
        t_emb = self.t_emb(time)
        hidden_states = self.hidden_emb(hidden_states) + t_emb
        context = self.context_emb(context)
        
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm_at1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm_at1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff1(self.norm_ff1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm_at2(hidden_states), context, deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward 2
        residual = hidden_states
        hidden_states = self.ff2(self.norm_ff2(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual


        # cross attention 3
        # residual = hidden_states
        # hidden_states = self.attn3(self.norm_at3(hidden_states), context, deterministic=deterministic)
        # hidden_states = hidden_states + residual

        # # feed forward last
        # residual = hidden_states
        # hidden_states = self.ff3(self.norm_ff3(hidden_states), deterministic=deterministic)
        # hidden_states = hidden_states + residual

        hidden_states = self.out(hidden_states)

        return hidden_states

class FlaxDenoisingBlock(nn.Module):
    """
    Flax Denoising Block
    """

    # internal dimension for transformer block
    dim: int
    out_dim: int
    n_heads: int
    d_head: int
    n_blocks: int=6
    context_emb_dim: int=512
    dropout: float = 0.0
    only_self_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.t_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.hidden_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.context_emb = nn.Sequential([
            nn.Dense(self.context_emb_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])
        self.norm_cond = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        self.out = nn.Sequential([
            nn.LayerNorm(epsilon=1e-5, dtype=self.dtype),
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.out_dim, dtype=self.dtype),
        ])

        # nn.Dense(self.out_dim, dtype=self.dtype)
        for i in range(self.n_blocks):
            setattr(self, f"attn{i}", FlaxCrossAttention(self.dim, self.n_heads, self.d_head, self.dropout, dtype=self.dtype))
            setattr(self, f"norm_at{i}", nn.LayerNorm(epsilon=1e-5, dtype=self.dtype))
            setattr(self, f"ff{i}", FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype))
            setattr(self, f"norm_ff{i}", nn.LayerNorm(epsilon=1e-5, dtype=self.dtype))

    def __call__(self, hidden_states, time, context, deterministic=False):
        # self attention
        t_emb = self.t_emb(time)
        hidden_states = self.hidden_emb(hidden_states) + t_emb
        context = self.norm_cond(self.context_emb(context))
        
        for i in range(self.n_blocks):
            if self.only_self_attention :
                context = hidden_states
            residual = hidden_states
            hidden_states = getattr(self, f"attn{i}")(
                getattr(self, f"norm_at{i}")(hidden_states), 
                context, 
                deterministic=deterministic
            )
            hidden_states = hidden_states + residual

            # feed forward
            residual = hidden_states
            hidden_states = getattr(self, f"ff{i}")(
                getattr(self, f"norm_ff{i}")(hidden_states),
                deterministic=deterministic
            )
            hidden_states = hidden_states + residual

        hidden_states = self.out(hidden_states)
        return hidden_states

class FlaxDenoisingBlockMLP(nn.Module):
    """
    Flax Denoising MLP Block
    """

    dim: int
    out_dim: int
    n_blocks: int = 6
    context_emb_dim: int = 512
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.t_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.hidden_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.context_emb = nn.Sequential([
            nn.Dense(self.context_emb_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.norm_cond = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        self.out = nn.Sequential([
            nn.LayerNorm(epsilon=1e-5, dtype=self.dtype),
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.out_dim, dtype=self.dtype),
        ])

        # Simple dense layers for the MLP style
        for i in range(self.n_blocks):
            setattr(self, f"mlp{i}", nn.Dense(self.dim, dtype=self.dtype))
            setattr(self, f"norm_mlp{i}", nn.LayerNorm(epsilon=1e-5, dtype=self.dtype))

    def __call__(self, hidden_states, time, context, deterministic=False):
        t_emb = self.t_emb(time)
        hidden_states = self.hidden_emb(hidden_states) + t_emb
        context = self.norm_cond(self.context_emb(context))
        hidden_states_im = jnp.concatenate([hidden_states, context], axis=-1)


        for i in range(self.n_blocks):
            residual = hidden_states
            hidden_states = getattr(self, f"mlp{i}")(
                getattr(self, f"norm_mlp{i}")(hidden_states_im)
            )
            hidden_states = nn.gelu(hidden_states)
            hidden_states = hidden_states + residual
            hidden_states_im = jnp.concatenate([hidden_states, context], axis=-1)

        hidden_states = self.out(hidden_states)
        return hidden_states

