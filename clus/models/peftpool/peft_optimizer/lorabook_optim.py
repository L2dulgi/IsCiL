from optax._src.transform import *
from optax._src.alias import * 
from jax import tree_util


def zero_grad_preserve(
    old_p,
    new_p,
    updates,
    eps=1e-6,
) : 
    """
    Combine two rank masks
    """
    def mask_fn(path, old, new, g):
        # print(path[-1])
        # aa = jnp.where(g==0, 1,0).astype(jnp.int32)
        # to = jnp.where(g==0,1,1).astype(jnp.int32)
        # print( jnp.sum(aa) / jnp.sum(to)    )
        return jnp.where(g==0, old, new)        
    return tree_util.tree_map_with_path(mask_fn, old_p, new_p, updates)

def scale_by_adam_lorabook(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  '''
  [Modified] update function only update the masked segment parameters
  '''
  
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)
  
  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    # [Modified] ================================= 
    mu = zero_grad_preserve(state.mu, mu, updates)
    nu = zero_grad_preserve(state.nu, nu, updates)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
   
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)

def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)

def adamw_lorapool(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  return combine.chain(
      scale_by_adam_lorabook(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )