import jax.random as random
import optax
from functools import partial

from clus.models.utils.utils import default, update_rngs
from clus.models.utils.train_state import create_train_state_time_cond
from clus.models.base.conditional_block import *
from clus.models.model.basic import BasicModel
from clus.models.base.mlp import CondMLP

import jax.numpy as jnp
import numpy as np

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float32) ** 2
        ) # original
        # betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float32)

    elif schedule == "cosine":
        timesteps = (
                np.arange(n_timestep + 1, dtype=np.float32) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float32)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float32) ** 0.5
    elif schedule == "vp":
        betas = vp_beta_schedule(n_timestep)
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def vp_beta_schedule(timesteps, dtype=np.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return np.asarray(betas, dtype=dtype)

def mse(a, b):
    def squared_error(a, b):
        return jnp.mean( jnp.square(a-b), axis=-1 )
    loss = squared_error(a,b)
    return loss

def noise_like(shape, rngs=None, repeat=False):
    if repeat:
        return random.normal(rngs, shape[1:])[jnp.newaxis].repeat(shape[0], axis=0)
    else:
        return random.normal(rngs, shape)

def extract_into_numpy(a, t, x_shape) :
    b = t.shape[0] # batch size
    out = jnp.take_along_axis(a, t, axis=-1) # shape of t
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # unsqueeze n times

class ConditionalDiffusion(BasicModel) :
    '''
    # basic conditional Diffusion training / evaluation module class
    '''
    def __init__(
        self,
        training = 'train', 
        model_config = None, 
        optimizer_config = None,
        input_config = None,
        out_dim = None,
        clip_denoised = True,
        diffusion_step = 64,
        **kwargs
    ) -> None:
        
        self.training = training 
        self.clip_denoised = clip_denoised
        
        self.input_config = input_config
        if input_config is None :
            self.input_config = {
                'hidden_states' : (1,1,4),
                'cond' : (1,1,60), # for kitchen dataset
            } # dict (shapes of input tensor shape for initialize parameters)

        self.dim_time_embedding = 128
        self.input_config['time'] = (1,1,self.dim_time_embedding)
    
        self.out_dim = out_dim
        if self.out_dim is None :
            self.out_dim = self.input_config['hidden_states'][-1]
            # self.out_dim = self.input_config['x'][-1]
        
        self.model_config = model_config
        if model_config is None :
            model_config = {
                'model_cls' :FlaxTimeCondTransformerBlock,
                'model_kwargs' : {
                    'dim' : 512,
                    'out_dim' : self.out_dim,
                    'n_heads' : 4,
                    'd_head' : 128,
                    'dropout' : 0.1,
                    'only_cross_attention' : True,
                }
            } # default model config 
            self.model_config = model_config
        
        self.model_config['model_kwargs']['out_dim'] = self.out_dim

        # evaluation model initialization 
        self.model_config['eval_kwargs'] = self.model_config['model_kwargs'].copy()
        self.model_config['eval_kwargs']['dropout'] = 0.0


        self.model = model_config['model_cls'](**model_config['model_kwargs'])
        self.model_eval = model_config['model_cls'](**model_config['eval_kwargs'])
        
        ### train state initialization ### 
        self.optimizer_config = optimizer_config
        if optimizer_config is None :
            self.optimizer_config = {
                'optimizer_cls' : optax.adam,
                'optimizer_kwargs' : {
                    'learning_rate' : 1e-4 ,
                    'b1' : 0.9,
                },
            } # example optimizer config 

        self.train_state = create_train_state_time_cond(
            self.model, 
            self.input_config,
            self.optimizer_config,
        )

        ## schedule init ## 
        self.schedule_time = diffusion_step
        self.v_posterior = 0.
        self.parameterization = 'eps'
        self.register_schedule(
            # beta_schedule='vp', 
            beta_schedule='linear', 
            # beta_schedule='cosine', 
            timesteps=self.schedule_time, 
            linear_end=1e-0 , # original 2e-2
        ) 
        
        ## sample rngs ##
        seed = 777
        self.sample_rngs = { 
            'p_noise' : random.PRNGKey(seed-2),
            'q_noise' : random.PRNGKey(seed-1),
            'apply' : random.PRNGKey(seed),
            'dropout' : random.PRNGKey(seed+1),
        }
        self.eval_rng_key = random.PRNGKey(seed+1)

        ## loss init info ##
        self.l_simple_weight = 1.
        self.original_elbo_weight = 0.

        ## log info ##
        self.log_every_t = 100

        ## for debug ##
        print( "==================== configurations ======================" )
        print( self.model_config ) 
        print( self.optimizer_config )
        print( self.input_config )
        print( "==========================================================" )

    def reinit_optimizer(self):
        old_params = self.train_state.params
        self.train_state = create_train_state_time_cond(
            self.model, 
            self.input_config,
            self.optimizer_config,
        )
        self.train_state = self.train_state.replace(
            params=old_params,
        )
    ### var scheduler ### 
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=10,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3) :
        
        betas = make_beta_schedule(beta_schedule , timesteps)

        alphas = 1. - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.append(1., alphas_cumprod[:-1])
        
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.betas= betas
        self.alphas_cumprod= alphas_cumprod
        self.alphas_cumprod_prev= alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod= jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod= jnp.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod= jnp.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod= jnp.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod= jnp.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1. - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance= posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped= jnp.log(jnp.clip(posterior_variance, a_min=1e-20))
        self.posterior_mean_coef1= betas * jnp.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2= (1. - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1. - alphas_cumprod)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * alphas) * (1 - self.alphas_cumprod)  # original 
            # lvlb_weights = self.betas ** 2 / ( (2 * self.posterior_variance * alphas) * (1 - self.alphas_cumprod) + 1e-10) # epsilon added
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * jnp.sqrt(alphas_cumprod) / (2. * 1 - (alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        
        lvlb_weights = np.array(lvlb_weights)
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights= jnp.array(lvlb_weights)

    ### loss related ###
    def get_loss(self, pred, target, mean=None) :
        return mse(pred, target)

    ### backward process ###
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_numpy(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_numpy(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_numpy(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_numpy(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_numpy(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_numpy(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_numpy(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_numpy(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_numpy(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @partial(jax.jit, static_argnums=(0,))
    def p_mean_variance(self, params, x, t, cond, return_model_out=False):    
        t_input = jax.lax.convert_element_type(t, jnp.float32)[:,jnp.newaxis]
        t_input = timestep_embedding(t_input, self.dim_time_embedding)

        model_out = self.model_eval.apply(params, x, t_input, cond, deterministic=True) 

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if self.clip_denoised:
            x_recon = jnp.clip( x_recon , -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        if return_model_out:
            return model_mean, posterior_variance, posterior_log_variance, model_out
        
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, params, x, t, cond, rngs=None, repeat_noise=False, return_model_out=False) :
        b = x.shape[0]
        if return_model_out :
            model_mean, _, model_log_variance, model_out = self.p_mean_variance(params, x=x, t=t, cond=cond, return_model_out=True)
        else : 
            model_mean, _, model_log_variance = self.p_mean_variance(params, x=x, t=t, cond=cond)
        noise = default( repeat_noise , lambda : jax.random.normal(rngs, x.shape) )
        # no noise when t == 0
        nonzero_mask = jnp.reshape(1 - jnp.equal(t, 0).astype(jnp.float32), (b,) + (1,) * (x.ndim - 1))

        if return_model_out == True :
            return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise, model_out
        
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, params, out_noise, cond, rngs=None, return_intermediates=False) :
        b = cond.shape[0] 
        out = out_noise 
        intermediates = [out]
        predictions = []
        # for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
        for i in reversed(range(0, self.num_timesteps)):
            out = self.p_sample(
                params, 
                out, 
                jnp.full((b,), i, dtype=jnp.int32),
                cond,
                rngs=rngs,
                return_model_out=return_intermediates,
            )
            rngs , _ = jax.random.split(rngs)
            # if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
            if return_intermediates:
                out, pred = out
                intermediates.append(out)
                predictions.append(pred)
        if return_intermediates: # NOTE
            return out, intermediates, predictions
        return out

    def p_losses(self, params, state, x_start, t, cond, noise=None, rngs=None) :
        noise = default(noise, lambda: jax.random.normal(key=rngs['p_noise'] ,shape=x_start.shape))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, rngs=rngs)
        t_input = jax.lax.convert_element_type(t, jnp.float32)[:,jnp.newaxis]
        t_input = timestep_embedding(t_input, self.dim_time_embedding)
        r1 , r2 = jax.random.split(rngs['apply'])
        apply_rng = {'params': r1, 'dropout': r2}
        model_out = state.apply_fn(params, x_noisy, t_input, cond, deterministic=False, rngs=apply_rng) 

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        
        # loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3]) # original embedding is dependon (c, h, w) image formulation
        loss = self.get_loss(model_out, target, mean=False).mean(axis=[-1]) # calculate mean value by data

        # embedding x_start (B, F) / pred (B, F) => each forwarded x_noizy is depend by T

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = jnp.mean( (self.lvlb_weights[t] * loss) ) # DDMP eq(12)
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    ### forward process ###
    @partial(jax.jit, static_argnums=(0,))
    def q_sample(self, x_start, t, noise=None, rngs=None) :
        noise = default(noise, lambda: jax.random.normal(rngs['q_noise'],shape=x_start.shape) ) # gaussian noise
        x_noise = ( extract_into_numpy(self.sqrt_alphas_cumprod,t, x_start.shape) * x_start + 
                   extract_into_numpy(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape) * noise )
        return x_noise

    ### jitted algos ### 
    def eval_model_out(self, x, t=None, cond=None) :
        @jax.jit
        def get_model_eval(params, x, t, cond) :
            t_input = jax.lax.convert_element_type(t, jnp.float32)[:,jnp.newaxis]
            t_input = timestep_embedding(t_input, self.dim_time_embedding)
            model_out = self.model_eval.apply(params, x, t_input, cond, deterministic=True) 
            return model_out
        
        noise = np.random.randn(*(x.shape[0],1,self.out_dim)) # shape (Batch, 1, F)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise, rngs=self.sample_rngs)
        model_out = get_model_eval(self.train_state.params, x_noisy, t, cond)      
        return model_out, t

    @partial(jax.jit, static_argnums=(0,))
    def train_model_jit(self, state, x, t, cond, noise, rngs=None):
        grad_fn = jax.grad(self.p_losses, has_aux=True)
        grads, loss_dict = grad_fn(state.params, state, x, t, cond, noise, rngs=rngs)
        metric = (None, loss_dict)

        state = state.apply_gradients(grads=grads)

        return state, metric

    def train_model(self, x, cond, t=None) :
        if x.ndim == 2 :
            x = x[:,None,:]
        if cond.ndim == 2 :
            cond = cond[:,None,:]

        if t is None :
            t = np.random.randint(0, self.num_timesteps, (x.shape[0],) ) 
        else : # shape must (Batch,)
            assert t.shape[0] == x.shape[0] , "t and x must have same batch size"
        noise = np.random.randn(*(x.shape[0],1,self.out_dim)) # shape (Batch, 1, F)
        self.train_state , metric = self.train_model_jit(self.train_state, x, t, cond, noise, rngs=self.sample_rngs)
        self.sample_rngs = update_rngs(self.sample_rngs)
        return metric
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_model_jit(self, params, out_noise, cond, rngs=None):
        out = self.p_sample_loop(params, out_noise, cond, rngs, return_intermediates=False)
        return out 
    
    # wrapper function for the
    def forward(self, **kwargs) :
        return self.p_sample_loop(**kwargs)
    
    def eval_model(self, cond, params=None):
        if cond.ndim == 2 :
            cond = cond[:,None,:]  
        out_noise = np.random.randn(*(cond.shape[0],1,self.out_dim))       
        self.eval_rng_key, rngs  = random.split(self.eval_rng_key)
        out = self.eval_model_jit(
                self.train_state.params if params is None else params, 
                out_noise, 
                cond, 
                rngs, 
            )
        return out
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_intermediates_jit(self, state, out_noise, cond, rngs=None):
        out , intermediates, pred = self.p_sample_loop(state.params, out_noise, cond, rngs, return_intermediates=True)
        return intermediates, pred
    
    def eval_intermediates(self, cond):
        # for same noise feature
        out_noise = np.random.randn(*(1,1,self.out_dim))      
        out_noise = np.repeat(out_noise, [cond.shape[0]], axis=0) 
        
        self.eval_rng_key, rngs  = random.split(self.eval_rng_key)
        intermediate , pred = self.eval_intermediates_jit(
                self.train_state, 
                out_noise, 
                cond, 
                rngs, 
            )
        return intermediate, pred


