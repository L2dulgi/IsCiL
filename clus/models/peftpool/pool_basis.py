import jax.numpy as jnp
from flax import linen as nn

## key similarity module ## 
class CosineSimilarityModule(nn.Module):
    '''
    Cosine Similarity Module for Query-Key matching 
    '''
    num_keys: int
    feature_dim: int
    
    def setup(self):
        self.keys = self.param(
            'keys',  
            nn.initializers.uniform(),
            (self.num_keys, self.feature_dim),
        )

    def __call__(self, queries):
        keys_norm = jnp.linalg.norm(self.keys, axis=-1, keepdims=True)
        queries_norm = jnp.linalg.norm(queries, axis=-1, keepdims=True)

        keys_normalized = self.keys / keys_norm
        queries_normalized = queries / queries_norm

        cos_sim = jnp.dot(queries_normalized, keys_normalized.T)    
        # for normalized range
        cos_sim = (cos_sim + 1) / 2
        return cos_sim
    

class ContrastiveSimilarityModule(nn.Module):
    '''
    Cosine Similarity Module for Query-Key matching 
    '''
    num_keys: int
    feature_dim: int
    
    def setup(self):
        # only updated by outside
        self.scaler = self.param(
            'scaler',  
            nn.initializers.uniform(),
            (self.num_keys, self.feature_dim),
            trainable=False
        )
    
        self.keys = self.param(
            'keys',  
            nn.initializers.uniform(),
            (self.num_keys, self.feature_dim),
        )

    def __call__(self, queries):
        keys_filtered = queries[...,None,:] * self.scaler[None,:,:] # ( B,1,f) , (1,k,f)
        keys_norm = jnp.linalg.norm(keys_filtered, axis=-1, keepdims=True) #( b,k,f)
        queries_norm = jnp.linalg.norm(queries, axis=-1, keepdims=True) # (k,f)

        keys_normalized = keys_filtered / keys_norm
        queries_normalized = queries / queries_norm

        # cos_sim = jnp.dot(queries_normalized, keys_normalized.T)    
        cos_sim =  jnp.sum( keys_normalized * queries_normalized, axis=-1) # (b,k)
        # for normalized range
        cos_sim = (cos_sim + 1) / 2
        return cos_sim

class MultiKeySimilarityModule(nn.Module):
    '''
    ### for multi-key matching algorithm
    Similarity query module for multi-key matching algorithm
    '''
    num_pool: int # max size of the pool
    key_num: int # key per number of pool
    feature_dim: int

    def setup(self):
        self.keys = self.param(
            'keys',  
            nn.initializers.uniform(),
            (self.num_pool, self.key_num, self.feature_dim),
        )

    def __call__(self, queries):
        '''
        queries : (batch_size, feature_dim)
        '''
        keys_norm = jnp.linalg.norm(self.keys, axis=-1, keepdims=True)
        queries_norm = jnp.linalg.norm(queries, axis=-1, keepdims=True)

        keys_normalized = self.keys / keys_norm
        queries_normalized = queries / queries_norm

        # cos_sim = jnp.dot(queries_normalized, keys_normalized.T)    
        cos_sim_total = jnp.tensordot(queries_normalized, keys_normalized, axes=([-1], [-1]))
        # selection method can be vary
        # max, mean, median etc.
        cos_sim = jnp.max(cos_sim_total, axis=-1) 

        # for normalized range
        cos_sim = (cos_sim + 1) / 2
        return cos_sim
 
def max_cosine_similarity_loss(params, state, query, balancing_prob, rngs=None):
    cos_sim = state.apply_fn(params, query)
    cos_sim = cos_sim * balancing_prob
    max_values = jnp.max(cos_sim, axis=-1) / cos_sim.shape[-1]
    loss = -jnp.mean(max_values)
    return loss, loss


class LoRAMemoryPoolConfig():
    def __init__(
        self,
        pool_length=10,
        feature_dim=512,
        action_dim=4,
        lora_dim=64,
        embedding_key="mean",
        prompt_init=nn.initializers.uniform(),
        prompt_pool=False,
        prompt_key=False,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="zero",
        num_classes_per_task=-1,
        num_layers=1,
        use_prefix_tune_for_e_prompt=False,
        num_heads=-1,
        same_key_value=False,
        key_num=1,
        meta_init_mode = 'copy',
        learned_processing = 'basic',
        eval_process = 'basic',
        ref_dropout = 0.0, 
        train_nested_dropout = False,
        consistency_mode = 'basic',
        tight_threshold = False,
    ) -> None:
        self.pool_length = pool_length
        self.feature_dim = feature_dim
        self.lora_dim = lora_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_pool = prompt_pool
        self.prompt_key = prompt_key
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key_init = prompt_key_init
        self.num_classes_per_task = num_classes_per_task
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.key_num = key_num
        self.meta_init_mode = meta_init_mode
        self.learned_processing = learned_processing
        # dylorabook
        self.ref_dropout = ref_dropout
        self.train_nested_dropout = train_nested_dropout
        self.action_dim = action_dim
        self.eval_process = eval_process
        self.consistency_mode = consistency_mode
        self.tight_threshold = tight_threshold

import numpy as np
from einops import rearrange
if __name__ =='__main__' :

    B,f,P,K = 10, 512, 21, 5
    b2 = 11
    q = np.random.randn(B, 11, f)
    k = np.random.randn(P, K, f)

    sim = np.tensordot(q, k, axes=([-1], [-1])) # B... P...
    sim = np.max(sim, axis=-1) # B P
    print(sim.shape)
