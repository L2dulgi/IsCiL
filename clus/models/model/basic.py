from transformers.models.gpt2.modeling_flax_gpt2 import * 

import jax.random as random
import optax
from functools import partial

from clus.models.utils.utils import default, update_rngs
from clus.models.base.conditional_block import *
from clus.models.base.mlp import MLP, NormalMLP
from clus.models.utils.loss import mse
from clus.models.utils.train_state import *
import cloudpickle
import pickle

class BasicModel() :
    '''
    # basic module class using flax # 
    [included]
    # flax model class 
    # train state(modle params, optimizer)
    # train algorithm, loss function 
    # evaluation metric, forward module 
    '''
    def __init__(self, mode) :
        '''
        mode : str, 'jit' or 'debug'
        '''
        self.mode = mode 
    
    def create_train_state(self) :
        ### if needed make custom function in module
        return None
    
    def reinit_optimizer(self) :
        old_params = self.train_state.params
        self.train_state = create_train_state_basic(self.model, self.input_config, self.optimizer_config)
        self.train_state.replace(params=old_params)
    
    def forward(self) : # Function for the flax calculation
        raise NotImplementedError

    def train_model(self) :
        raise NotImplementedError
    
    def eval_model(self) :
        raise NotImplementedError
    
    def save_model(
            self,
            model_path:str,
            options:dict=None,
        ) :
        if options['full_saving'] == True :
            with open(model_path, 'wb') as f:
                cloudpickle.dump(self, f)
        else :
            with open(model_path, 'wb') as f:
                cloudpickle.dump(self.train_state.params, f)
        
    
    def load_model(
            self,
            model_path:str,
            options:dict=None,
        ) :
        pass
    
class MLPModule(BasicModel) :
    '''
    # mlp module class using flax #
    '''
    def __init__(
            self,
            mode = 'train',
            model_config=None,
            input_config=None,
            optimizer_config=None,
            shtochastic=False,
    ) -> None : 
        super().__init__(mode=mode)

        # TEMP 
        self.model_config ={
            'hidden_size' : 512,
            'out_shape' : 9,
            'num_hidden_layers' : 4,
            'dropout' : 0.0,
        }
        if model_config is not None :
            self.model_config = model_config
        
        self.stochastic = shtochastic
        if self.stochastic is True :
            self.model_config['out_shape'] *= 2 

        # self.model_config = model_config if model_config is None else self.model_config
        self.input_config = input_config
        self.optimizer_config = optimizer_config
        if optimizer_config is None :
            self.optimizer_config = {
                'optimizer_cls' : optax.adam,
                'optimizer_kwargs' : {
                    'learning_rate' : 5e-5,
                    'b1' : 0.9,
                },
            }
        ## sample rngs ##
        seed = 777
        self.sample_rngs = { 
            'p_noise' : random.PRNGKey(seed-2),
            'q_noise' : random.PRNGKey(seed-1),
            'apply' : random.PRNGKey(seed),
            'dropout' : random.PRNGKey(seed+99),
        }
        self.eval_rng_key = random.PRNGKey(seed+1)


        self.model = MLP(
            hidden_size=self.model_config['hidden_size'],
            out_shape=self.model_config['out_shape'],
            dropout_rate=0.0, 
            deterministic=True
        )
        self.model_eval = MLP(
            hidden_size=self.model_config['hidden_size'],
            out_shape=self.model_config['out_shape'], 
            dropout_rate=0.0, 
            deterministic=True
        )

        self.train_state = create_train_state_basic(
            self.model, 
            input_config=self.input_config,
            optimizer_config=self.optimizer_config
        )

    def forward(self, params, x, rngs=None) :
        '''
        wrapper function for model apply function
        '''
        out = self.model.apply(params, x, rngs=rngs)
        return out

    def loss_fn(self, params, state, batch, rngs=None):
        logits = state.apply_fn(params, batch['inputs'], rngs=rngs)
        loss = mse(logits, batch['labels'])
        return loss , None

    @partial(jax.jit, static_argnums=(0,))
    def train_model_jit(self, state, batch, rngs=None):
        grad_fn = jax.grad(self.loss_fn, has_aux=True)
        grads , _= grad_fn(state.params, state, batch, rngs=rngs)
        state = state.apply_gradients(grads=grads)

        metric , _ = self.loss_fn(state.params, state, batch, rngs=rngs)
        return state, metric

    def train_model(self, batch):
        self.train_state , metric = self.train_model_jit( self.train_state, batch, rngs=self.sample_rngs)
        self.sample_rngs = update_rngs(self.sample_rngs)
        return metric
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_model_jit(self, state, x, rngs=None):
        out = self.model_eval.apply(state.params, x, rngs=rngs)
        return out 

    def eval_model(self, x):
        self.eval_rng_key, rngs  = random.split(self.eval_rng_key)
        out = self.eval_model_jit(self.train_state,x)
        return out