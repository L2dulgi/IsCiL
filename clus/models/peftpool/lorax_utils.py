

from lorax.helpers import *
from lorax.transform import *
import lorax

##### LoRA pool class and functions ######
# [v] LoraWeightPool - base implict array
# [v] init_lora_pool - init lora pool
# [v] warp_pool_optimizer - warp optimizer with pool mask
# [v] set_pool_mask - set pool mask 
@dataclass
class LoraWeightPool(qax.ImplicitArray):
    '''
    lora with masked path
    '''
    w : qax.ArrayValue # M x N

    pool_mask : qax.ArrayValue # B ( zero one value )
    a : qax.ArrayValue # B x k x N
    b : qax.ArrayValue # B x M x k
    alpha : float = qax.aux_field(default=1.)

    def __post_init__(self):
        super().__post_init__()
        assert self.a.shape[-2] == self.b.shape[-1]

    def materialize(self):
        lora_pool = self.get_scale() * self.b @ self.a
        masked_lora_pool = self.pool_mask[..., None, None] * lora_pool 
        delta_w = jnp.sum(masked_lora_pool, axis=0) / jnp.sum(self.pool_mask)
        return (self.w + delta_w).astype(self.w.dtype)

    def get_scale(self):
        return self.alpha / self.b.shape[-1]

@dataclass
class DualLoraWeightPool(qax.ImplicitArray):
    '''
    lora with masked path
    '''
    w : qax.ArrayValue # M x N

    pool_mask_g : qax.ArrayValue # B ( zero one value )
    pool_mask_t : qax.ArrayValue # B ( zero one value )
    a_g : qax.ArrayValue # B x k x N
    b_g : qax.ArrayValue # B x k x N
    a_t : qax.ArrayValue # B x k x N
    b_t : qax.ArrayValue # B x M x k
    alpha : float = qax.aux_field(default=1.)
    
    def __post_init__(self):
        super().__post_init__()
        assert self.a.shape[-2] == self.b.shape[-1]

    def materialize(self):
        a_g = jnp.mean(self.pool_mask_g[..., None, None] * self.a_g, axis=0)
        b_g = jnp.mean(self.pool_mask_g[..., None, None] * self.b_g, axis=0 )
        a_t = jnp.mean(self.pool_mask[..., None, None] * self.a_t, axis=0)
        b_t = jnp.mean(self.pool_mask[..., None, None] * self.b_t, axis=0)
        a = a_g + a_t
        b = b_g + b_t
        lora_w = self.get_scale() * b @ a
        return (self.w + lora_w).astype(self.w.dtype)

    def get_scale(self):
        return self.alpha / self.b.shape[-1]

@dataclass
class LoRABookWeight(qax.ImplicitArray):
    '''
    lora with masked path
    '''
    w : qax.ArrayValue # M x N
    pool_mask : qax.ArrayValue # B (zero one values for implementation.)
    a : qax.ArrayValue # Book x N
    b : qax.ArrayValue # M x Book
    alpha : float = qax.aux_field(default=1.)

    def __post_init__(self):
        super().__post_init__()
        # assert self.a.shape[-2] == self.b.shape[-1]

    def materialize(self):
        a = self.a * self.pool_mask[..., None] # Book N
        b = self.b * self.pool_mask[None, ...] # M Book
        
        lora_codes = self.get_scale() * jnp.dot( b, a)
        return (self.w + lora_codes).astype(self.w.dtype)

    def get_scale(self):
        return self.alpha / jnp.sum(self.pool_mask)


def init_lora_pool(
        param_tree, 
        spec, 
        rng,
        stddev=0.01, 
        dtype=jnp.float32, 
        alpha=1., 
        pool_size=10, # default pool size
        is_leaf=None,
        mode = 'single', # single or dual 
    ):
    
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key
    key_it = iter_keys(rng)

    implict_array_cls = LoraWeightPool
    if mode == 'dual' :
        implict_array_cls = DualLoraWeightPool

    def get_param(path, param, spec_val):
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            pool_mask = jnp.zeros((pool_size,), dtype=dtype)
            a = jax.random.normal(next(key_it), (pool_size, spec_val, a_dim), dtype=dtype) * stddev
            b = jnp.zeros((pool_size, b_dim, spec_val), dtype=dtype)

            implict_array_kwargs = {
                'w' : param,
                'pool_mask' : pool_mask,
                'alpha' : alpha,
            }
            if mode =='single' : 
                implict_array_kwargs['a'] = a
                implict_array_kwargs['b'] = b
            if mode == 'dual' :
                a_g = jax.random.normal(next(key_it), (spec_val*2, a_dim), dtype=dtype) * stddev
                b_g = jnp.zeros((b_dim, spec_val*2), dtype=dtype)
                implict_array_kwargs['a_g'] = a_g
                implict_array_kwargs['b_g'] = b_g
                implict_array_kwargs['pool_mask_g'] = jnp.zeros((pool_size,), dtype=dtype)
                implict_array_kwargs['a_t'] = a
                implict_array_kwargs['b_t'] = b


            return implict_array_cls(**implict_array_kwargs)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        pool_mask = jnp.zeros((pool_size,), dtype=dtype)
        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev

        a = jnp.repeat(a, pool_size, axis=0)
        b = jnp.repeat(b, pool_size, axis=0)

        implict_array_kwargs = {
            'w' : param,
            'pool_mask' : pool_mask,
            'alpha' : alpha,
        }

        if mode =='single' : 
            implict_array_kwargs['a'] = a
            implict_array_kwargs['b'] = b
        if mode == 'dual' :
            a_g = jnp.zeros((
                *(1 for _ in range(len(window_shape))),
                spec_val*2,
                out_channels
            ), dtype=param.dtype)
            b_g = jax.random.normal(rng, (*window_shape, in_channels, spec_val*2), dtype=param.dtype) * stddev
            implict_array_kwargs['a_g'] = a_g
            implict_array_kwargs['b_g'] = b_g
            implict_array_kwargs['pool_mask_g'] = jnp.zeros((pool_size,), dtype=dtype)
            implict_array_kwargs['a_t'] = a
            implict_array_kwargs['b_t'] = b
        
        return implict_array_cls(param, pool_mask, a, b, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)

def wrap_pool_optimizer(
        optimizer : optax.GradientTransformation, 
        spec, 
        scalar_frozen_grads=False, 
        mode='single',
        fixed_components = ['w', 'pool_mask'],
        pool_cls = None,
    ):
    print( f'[wrap_optimizer] wrap_pool_optimizer : mode {mode}')
    full_freeze_labels = jax.tree_map(
        lambda x: 'freeze' if x == LORA_FREEZE else 'train',
        spec
    )
    optimizer_with_full_freeze = qax.utils.freeze_subtrees(
        optimizer,
        full_freeze_labels,
        use_scalar_zeros=scalar_frozen_grads
    )
    if pool_cls is None :
        pool_cls = LoraWeightPool if mode == 'single' else DualLoraWeightPool
    return qax.freeze_keys(optimizer_with_full_freeze, pool_cls, fixed_components, use_scalar_zeros=scalar_frozen_grads)

def set_pool_mask(params, mask, mode='t'):
    '''
    pool_mask setter
    * must used in LoraWeightPool's masking function
    '''
    mask_postfix = '' if mode == 't' else f"_{mode}"
    target_mask = f'pool_mask{mask_postfix}'
    def set_mask_leaf(path, param):
        if path[-1] == target_mask:
            if param.shape != mask.shape:
                raise ValueError(f'mask shape must be equal to param mask shape {param.shape}, but got {mask.shape}\n in path {path}')
            return mask
        return param
    return jax.tree_util.tree_map_with_path(set_mask_leaf, params)

#### LoRABook pool class and functions ######
def init_lora_book(
        param_tree, 
        spec, 
        rng, 
        stddev=0.01, 
        dtype=jnp.float32, 
        alpha=1., 
        is_leaf=None,
        book_size = 128,
    ):
    '''
    actually sepc_val is not used.
    '''
    print( f'[lora_book] init_lora_book : book_size {book_size}')
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(rng)

    def get_param(path, param, spec_val):
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape
            pool_mask = jnp.zeros((book_size,), dtype=dtype)
            b = jnp.zeros((b_dim, book_size), dtype=dtype)
            a = jax.random.normal(next(key_it), (book_size, a_dim), dtype=dtype) * stddev
            return LoRABookWeight(w=param, pool_mask=pool_mask, a=a, b=b, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        pool_mask = jnp.zeros((book_size,), dtype=dtype)
        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            book_size,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, book_size), dtype=param.dtype) * stddev
        return LoRABookWeight(param, pool_mask=pool_mask, a=a, b=b, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)



##### NOLA pool class and functions ######
@dataclass
class NOLAWeightPool(qax.ImplicitArray):
    '''
    lora with masked path
    '''
    w : qax.ArrayValue # M x N
    # alphas in NOLA paper (B,)
    nola_a : qax.ArrayValue 
    nola_b : qax.ArrayValue 
    # random params
    # TODO change this by PRNG generation by fixed seed
    a : qax.ArrayValue # B x r x N
    b : qax.ArrayValue # B x M x r
    alpha : float = qax.aux_field(default=1.)

    def __post_init__(self):
        super().__post_init__()
        assert self.a.shape[-2] == self.b.shape[-1]

    def materialize(self) :
        nola_a = self.nola_a[..., None, None]
        nola_b = self.nola_b[..., None, None]
        nola_a = jnp.sum(nola_a * self.a, axis=0)
        nola_b = jnp.sum(nola_b * self.b, axis=0)
        nola_weight = self.get_scale() * nola_b @ nola_a
        return (self.w + nola_weight).astype(self.w.dtype)

    def get_scale(self):
        return self.alpha / self.b.shape[-1]

def init_nola_pool(
        param_tree, 
        spec, 
        rng,
        stddev=0.01, # default to 0.01
        dtype=jnp.float32, 
        alpha=1., 
        pool_size=10, # default pool size
        is_leaf=None,
    ):
    k = 256
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key
    key_it = iter_keys(rng)

    implict_array_cls = NOLAWeightPool

    def get_param(path, param, spec_val):
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            nola_a = jax.random.normal( next(key_it),(k,), dtype=dtype)
            nola_b = jax.random.normal( next(key_it),(k,), dtype=dtype)
            # nola is all normal distribution
            a = jax.random.normal(next(key_it), (k, spec_val, a_dim), dtype=dtype) * stddev
            b = jax.random.normal(next(key_it), (k, b_dim, spec_val), dtype=dtype) * stddev
            implict_array_kwargs = {
                'w' : param,
                'nola_a' : nola_a,
                'nola_b' : nola_b,
                'alpha' : alpha,
            }
            implict_array_kwargs['a'] = a
            implict_array_kwargs['b'] = b


            return implict_array_cls(**implict_array_kwargs)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        nola_a = jax.random.normal( key_it,(pool_size,), dtype=dtype)
        nola_b = jax.random.normal( next(key_it),(pool_size,), dtype=dtype)
        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev

        a = jnp.repeat(a, pool_size, axis=0)
        b = jnp.repeat(b, pool_size, axis=0)

        implict_array_kwargs = {
            'w' : param,
            'nola_a' : nola_a,
            'nola_b' : nola_b,
            'alpha' : alpha,
        }

        implict_array_kwargs['a'] = a
        implict_array_kwargs['b'] = b
        # not implemented yet
        return 0
    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)

def wrap_nola_optimizer(
        optimizer : optax.GradientTransformation, 
        spec, 
        scalar_frozen_grads=False, 
        mode='single',
    ):
    full_freeze_labels = jax.tree_map(
        lambda x: 'freeze' if x == LORA_FREEZE else 'train',
        spec
    )
    optimizer_with_full_freeze = qax.utils.freeze_subtrees(
        optimizer,
        full_freeze_labels,
        use_scalar_zeros=scalar_frozen_grads
    )
    
    pool_cls = LoraWeightPool if mode == 'single' else DualLoraWeightPool
    return qax.freeze_keys(optimizer_with_full_freeze, pool_cls, ['w', 'a', 'b'], use_scalar_zeros=scalar_frozen_grads)

def set_nola_mask(params, mask, mode='t'):
    '''
    pool_mask setter
    * must used in LoraWeightPool's masking function
    '''
    return params
    mask_postfix = '' if mode == 't' else f"_{mode}"
    target_mask = f'pool_mask{mask_postfix}'
    def set_mask_leaf(path, param):
        if path[-1] == target_mask:
            if param.shape != mask.shape:
                raise ValueError(f'mask shape must be equal to param mask shape {param.shape}, but got {mask.shape}\n in path {path}')
            return mask
        return param
    return jax.tree_util.tree_map_with_path(set_mask_leaf, params)

# 1125 TODO integrate lora-family
class LoRAInitializer() :
    def __init__(self) -> None:
        pass

    ## core functions ##
    def init_lora_pool():
        pass

    def wrap_pool_optimizer():
        pass
    ## mode functions ##
