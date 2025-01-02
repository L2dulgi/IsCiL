import jax.numpy as jnp


def mse(a, b):
    error = a - b
    squared_error = jnp.square(error)
    mean_squared_error = jnp.mean(squared_error)
    return mean_squared_error

# similarity based loss function( contrastive loss)
def cossim_loss(a,b):
    def cos_sim(a, b):
        a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
        b = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        return jnp.sum(a * b, axis=-1)
    
    cos_sim_val = cos_sim(a,b)
    return jnp.mean( 1 - cos_sim_val)