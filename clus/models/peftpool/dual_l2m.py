import jax.numpy as jnp
from clus.models.peftpool.lorax_utils import *
import flax.linen as nn
import jax

from clus.models.peftpool.pool_basis import *
from clus.models.peftpool.lorax_utils import *
from clus.env.offline import MemoryPoolDataloader
from clus.models.model.cdm import ConditionalDiffusion
from flax.training import train_state
import numpy as np

from clus.models.peftpool.pool_basis import LoRAMemoryPoolConfig
from clus.utils.utils import *

from clus.models.peftpool.skill_hash import ConsistencyHashModel, ContextHashModel, DualHashModel, DualHashMultiKeyModel

# learning module of LoRA

class LoRAPoolModel():
    '''
    wrapper function for LoRAPool training 
    '''
    def __init__(
            self,
            model=None, 
            memory_pool_config=LoRAMemoryPoolConfig(
                pool_length=50,
                feature_dim=652, # mmworld 652 # kitchen 572
                lora_dim=4,
                embedding_key='split'
            ),
            key_module_cls=ContextHashModel, 
            lora_optimizer_config=None,
            context_mode='obs', 
            retrieval_mode='sim',
            key_mode = 'single',
            learnable_key=True,
            tail_flag=False,
        ) -> None:

        # set lora pool
        print('[lora pool model] context_mode : ', context_mode)
        print('[lora pool model] retrieval_mode : ', retrieval_mode)
        print('[lora pool model] optimizer_config : ', lora_optimizer_config)
        print('[lora pool model] tail_flag : ', tail_flag)  
        self.model = model # basic Module(model class)
        self.memory_pool_config = memory_pool_config # memory pool class
        self.key_module_cls = key_module_cls
        self.lora_optimizer_config = lora_optimizer_config
        self.tail_flag = tail_flag

        self.set_lorapool()
        if key_module_cls == ContextHashModel :
            print("[key_module] ContextHashModel")
            self.key_module = key_module_cls(
                memory_pool_config=self.memory_pool_config,
                key_mode = key_mode,
            )
        elif key_module_cls == DualHashModel :
            print("[key_module] DualHashModel")
            self.key_module = key_module_cls(
                memory_pool_config=self.memory_pool_config,
                context_dim=self.memory_pool_config.feature_dim,
                context_mode=context_mode,
                retrieval_mode=retrieval_mode,
            )
        elif key_module_cls == DualHashMultiKeyModel :
            print("[key_module] DualHashMultiKeyModel")
            print(self.memory_pool_config.feature_dim)
            self.key_module = key_module_cls(
                memory_pool_config=self.memory_pool_config,
                context_dim= self.memory_pool_config.feature_dim,
                context_mode=context_mode,
                retrieval_mode=retrieval_mode,
                learnable_key=learnable_key,
            )
        elif key_module_cls == ConsistencyHashModel :
            print("[key_module] ConsistencyHashModel")
            print(self.memory_pool_config.feature_dim)
            self.key_module = key_module_cls(
                memory_pool_config=self.memory_pool_config,
                context_dim= self.memory_pool_config.feature_dim,
                context_mode=context_mode,
                retrieval_mode=retrieval_mode,
                learnable_key=learnable_key,
            )
        else :
            raise Exception(f"[ERROR] key_module_cls must be ContextHashModel or DualHashModel or DualHashMultiKeyModel \nbut, {key_module_cls}")
      
    def set_lorapool(self) :
        '''
        set lora pool 
        '''
        print(f"[converting model to lora model ...]")
        # get basis of model
        train_state_orig = self.model.train_state
        # optimizer_orig = self.model.optimizer
        orig_params = train_state_orig.params

        # lora model setting 
        self.model.model.apply = lorax.lora(self.model.model.apply)
        self.model.model_eval.apply = lorax.lora(self.model.model_eval.apply)

        def decision_fn(path, param):
            if 'embedding' in path:
                print(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.memory_pool_config.lora_dim
            print(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        # tune_vectors = kernel or something training flag
        self.lora_spec = lorax.simple_spec(orig_params, decision_fn=decision_fn, tune_vectors=False)
        
        lora_params = init_lora_pool(
            orig_params, 
            self.lora_spec, 
            jax.random.PRNGKey(0),
            pool_size=self.memory_pool_config.pool_length,
        )
        self.init_lora_params = lora_params
        # nola test
        # lora_params = init_nola_pool(
        #     train_state_orig.params, 
        #     self.lora_spec, 
        #     jax.random.PRNGKey(0),
        #     pool_size=self.memory_pool_config.pool_length,
        # )

        if self.lora_optimizer_config is None :
            self.lora_optimizer_config ={
                'optimizer_cls' : optax.adam,
                'optimizer_kwargs' : {
                    'learning_rate' : 1e-4,
                    # 'weight_decay' : 1e-4,
                },
            }

        lora_optimizer = self.init_lora_optimizer()
        # create train state by lora params
        self.model.train_state = train_state.TrainState.create(
            apply_fn=self.model.model.apply, 
            params=lora_params, 
            tx=lora_optimizer,
        )

    def init_lora_optimizer(self) :
        optimizer_config = self.lora_optimizer_config   
        optimizer = optimizer_config['optimizer_cls'](**optimizer_config['optimizer_kwargs'])
        lora_optimizer = wrap_pool_optimizer( 
            optimizer,
            self.lora_spec,   
        )
        # lora_optimizer = wrap_nola_optimizer( 
        #     optimizer,
        #     self.lora_spec,   
        # )
        return lora_optimizer
    
    def set_pool_mask(self, pool_idx, eval=False) :
        pool_mask = np.zeros((self.memory_pool_config.pool_length,), jnp.float32)
        pool_mask[pool_idx] = 1. # NOTE : pool_idx is list of index
        replace_params = set_pool_mask(self.model.train_state.params, pool_mask)
        if eval : 
            return replace_params
        else : 
            self.model.train_state = self.model.train_state.replace(params=replace_params)
            return None
        
    #### main training module #### 
    def process_key_selection(self, dataloader, first_epoch=False):
        # Process the data differently based on the epoch and the key module class
        if first_epoch:
            # Process data for the first epoch
            dataloader.stacked_data = self.key_module.process_dataset_query(dataloader.stacked_data)
            if self.key_module_cls == ConsistencyHashModel :
                processed_data = self.key_module.process_dataset_key(
                    dataloader.stacked_data, 
                    model=self,
                    )
            else :
                processed_data = self.key_module.process_dataset_key(dataloader.stacked_data)
        elif self.key_module_cls == ContextHashModel and self.key_module.key_mode != 'CoLoR' \
            and self.key_module.key_mode != 'TAIL':
            # Process data for subsequent epochs with specific key module conditions
            processed_data = self.key_module.process_dataset_key(dataloader.stacked_data)
        else:
            # Skip processing if conditions are not met
            processed_data = dataloader.stacked_data

        # Unpack the processed data if it's a tuple
        if isinstance(processed_data, tuple):
            dataloader.stacked_data, _ = processed_data
        else:
            dataloader.stacked_data = processed_data

        # Update the key module with the processed data
        metric = self.key_module.update_model(dataloader.stacked_data)
        
        return dataloader

    def train_model(self, dataloader, batch_size=1024, first_epoch=False) :
        '''
        unlike the original model.
        memory pool model requires the full dataset for query-key pre-calculation and custom training loop
        ''' 
        dataloader = self.process_key_selection(dataloader, first_epoch=first_epoch)

        # traininig loop
        total_loss = 0
        for b_count, batch_tuple in enumerate(dataloader.get_all_batch(batch_size=batch_size, pool_key='key')) :
            pool_idx , batch = batch_tuple  
            
            self.set_pool_mask(pool_idx)
            if isinstance(self.model, ConditionalDiffusion) :
                cond = batch['observations']
                x = batch['actions']
                metric = self.model.train_model(x=x,cond=cond)
                loss = metric[1]['train/loss']
            else :
                input_batch = {
                    'inputs' : batch['observations'],
                    'labels' : batch['actions'],
                }
                loss = self.model.train_model(input_batch)
            total_loss += loss
        total_loss /= b_count + 1
        return total_loss   
    
    #### forward original model functions #### 
    def eval_model_single(self, obs, params=None) :
        query = self.key_module.process_dataset_query(obs)
        max_indicies , cos_sim = self.key_module.retrieve_key(query)

        task_mask = np.zeros((self.memory_pool_config.pool_length,), jnp.float32)
        task_mask[max_indicies] = 1.
        self.model.train_state = self.model.train_state.replace(params=set_pool_mask(self.model.train_state.params, task_mask))
        action = self.model.eval_model(obs[:,None,:], params=params)
        return action

    def eval_model(self, obs, prev_action=None, return_unique=False, task_id=None) : # multi evaluation function (core!)
        # obs is set to be (B, S, F)
        query_dict = {
            'observations' : obs.copy(),
            'prev_actions' : prev_action.copy() if prev_action is not None else None,
        }
        query = self.key_module.process_dataset_query(query_dict)
        if obs.ndim == 3 and obs.shape[1] > 1 :
            obs = obs[:,-1:,:]
        max_indicies , cos_sim = self.key_module.retrieve_key(query, task_id=task_id)
        if max_indicies.ndim == 0 :
            max_indicies = max_indicies[None]
        unique_indices = np.unique(max_indicies) 
        for pool_idx in unique_indices :
            p = self.set_pool_mask(pool_idx, eval=True)
            action = self.model.eval_model(obs, params=p)
            indicies_mask = (max_indicies == pool_idx).astype(int)[:,None]
            action = action * indicies_mask
            if pool_idx == unique_indices[0] :
                action_ret = action
            else :
                action_ret += action
        if return_unique :
            return action_ret, unique_indices
        return action_ret
    
    def eval_model_by_mask(self, obs, prev_action=None, return_unique=False) : # multi evaluation function (core!)
        # obs is set to be (B, S, F)
        '''
        [0110] used to evalutate the CoLoR expriments for aggregated evaluation
        '''
        query_dict = {
            'observations' : obs.copy(),
            'prev_actions' : prev_action.copy() if prev_action is not None else None,
        }
        query = self.key_module.process_dataset_query(query_dict)
        if obs.ndim == 3 and obs.shape[1] > 1 :
            obs = obs[:,-1:,:]
        eval_mask , cos_sim = self.key_module.retrieve_aux_key(query)

        for mid , mask in enumerate(eval_mask[:,0,:]) :
            replace_params = set_pool_mask(self.model.train_state.params, mask)
            action = self.model.eval_model(obs, params=replace_params) 
            if mid == 0 :
                action_ret = action
            else :
                action_ret = action_ret.at[mid].set(action[mid].copy())
        return action_ret
    
    #### utill functions #### 
    def reinit_optimizer(self) :
        lora_optimizer = self.init_lora_optimizer()
        self.model.train_state = train_state.TrainState.create(
            apply_fn=self.model.model.apply, 
            params=self.model.train_state.params.copy(), 
            tx=lora_optimizer,
        )

    def next_phase_processing(self, phase=None) :
        self.key_module.key_appear_count_fixed = self.key_module.key_appear_count.copy()

class CrossDomainLoRAPoolModel(LoRAPoolModel):
    def __init__(
            self,
            sample_input=None, # dictionary of input
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        
    def update_params(self, params_a, params_b):
        """
        Recursively matches the shapes of params in params_b with those in params_a.
        Pads extra elements with zeros.
        """
        result = {}

        for key in params_b:
            if key in params_a:
                if isinstance(params_b[key], dict):
                    # Recursive call if the value is a nested dictionary
                    result[key] = self.update_params(params_a[key], params_b[key])
                else:
                    # Perform the array matching and padding
                    a_val = jnp.array(params_a[key])
                    b_val = jnp.zeros_like(params_b[key])
                    min_shape = tuple(min(a, b) for a, b in zip(a_val.shape, b_val.shape))
                    slices = tuple(slice(0, m) for m in min_shape)
                    b_val = b_val.at[slices].set(a_val[slices])
                    result[key] = b_val
            else:
                # If the key is not in params_a, create a zero array of the same shape as in params_b
                result[key] = jnp.zeros_like(params_b[key])

        return result
    
    def set_lorapool(self) :
        '''
        set lora pool 
        '''
        print(f"[converting model to lora model ...]")
        # get basis of model
        train_state_orig = self.model.train_state

        ## modify the last layer of model and params shape ## 
        orig_model_cls = self.model.model.__class__
        orig_model_kwargs = vars(self.model.model)
        del(orig_model_kwargs['_parent_ref'])
        del(orig_model_kwargs['name'])
        del(orig_model_kwargs['_id'])
        del(orig_model_kwargs['_state'])
        self.model.model = orig_model_cls(**orig_model_kwargs)
        orig_model_kwargs['dropout'] = 0.
        self.model.model_eval = orig_model_cls(**orig_model_kwargs)

        orig_params = train_state_orig.params
        modified_params = self.model.model_eval.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1,1,4)),
            jnp.zeros((1,1,self.model.dim_time_embedding)),
            jnp.zeros((1,1,652)),
        )
        orig_params = self.update_params(orig_params, modified_params)

        # lora model setting 
        self.model.model.apply = lorax.lora(self.model.model.apply)
        self.model.model_eval.apply = lorax.lora(self.model.model_eval.apply)

        def decision_fn(path, param):
            if 'embedding' in path:
                print(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.memory_pool_config.lora_dim
            print(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        # tune_vectors = kernel or something training flag
        self.lora_spec = lorax.simple_spec(orig_params, decision_fn=decision_fn, tune_vectors=False)
        
        lora_params = init_lora_pool(
            orig_params, 
            self.lora_spec, 
            jax.random.PRNGKey(0),
            pool_size=self.memory_pool_config.pool_length,
        )
        self.init_lora_params = lora_params

        self.lora_optimizer_config ={
            'optimizer_cls' : optax.adamw,
            'optimizer_kwargs' : {
                'learning_rate' : 1e-4,
                'weight_decay' : 1e-4,
            },
        }

        lora_optimizer = self.init_lora_optimizer()
        # create train state by lora params
        self.model.train_state = train_state.TrainState.create(
            apply_fn=self.model.model.apply, 
            params=lora_params, 
            tx=lora_optimizer,
        )


from tqdm import tqdm
class PesudoReplayLoRAPoolModel(LoRAPoolModel) :
    '''
    wrapper function for LoRAPool training with seen context replay
    require : keymodule must have functionality for descriminate seen context
    '''

    #### main training module #### 
    def train_model(self, dataloader, batch_size=1024, first_epoch=False) :
        '''
        unlike the original model.
        memory pool model requires the full dataset for query-key pre-calculation and custom training loop
        ''' 
        if not isinstance(dataloader, MemoryPoolDataloader) :
            raise Exception(f"[ERROR] dataloader type must be MemoryPoolDataloader in LoRAModel \nbut, {type(dataloader)}")
        
        if first_epoch or self.key_module_cls == ContextHashModel :
            dataloader.stacked_data = self.key_module.process_dataset_query(dataloader.stacked_data)
            processed_data = self.key_module.process_dataset_key(dataloader.stacked_data)
            
            if type(processed_data) == tuple :
                dataloader.stacked_data, _ = processed_data
            else :
                dataloader.stacked_data = processed_data

            dataloader.stacked_data = self.process_pesudo_replay(dataloader.stacked_data)

        metric = self.key_module.update_model(dataloader.stacked_data)

        # traininig loop
        total_loss = 0
        for b_count, batch_tuple in enumerate(dataloader.get_all_batch(batch_size=batch_size, pool_key='key')) :
            pool_idx , batch = batch_tuple  
            # set model mask
            pool_mask = np.zeros((self.memory_pool_config.pool_length,), jnp.float32)
            pool_mask[pool_idx] = 1. # NOTE : pool_idx is list of index
            self.model.train_state = self.model.train_state.replace(params=set_pool_mask(self.model.train_state.params, pool_mask))
            if isinstance(self.model, ConditionalDiffusion) :
                cond = batch['observations']
                x = batch['actions']
                metric = self.model.train_model(x=x,cond=cond)
                loss = metric[1]['train/loss']
            else :
                input_batch = {
                    'inputs' : batch['observations'],
                    'labels' : batch['actions'],
                }
                loss = self.model.train_model(input_batch)
            total_loss += loss
        total_loss /= b_count
        return total_loss   

    #### utill functions ####
    def next_phase_processing(self):
        super().next_phase_processing()
        self.seen_contexts = self.key_module.get_seen_context_idx()

    def process_pesudo_replay(self, dataset) :
        unique_keys = np.unique(dataset['key'])
        seen_contexts = self.key_module.get_seen_context_idx()

        print(f'[pesudo replay] prev dataset count : ', dataset['observations'].shape)
        aug_dataset_total = None
        # process
        process_batch = 1024 
        for key in tqdm(unique_keys) :
            if key in seen_contexts :
                # extend pseudo replay data from initial modulator
                aug_dataset = {
                    k : v[dataset['key']==key].copy() for k, v in dataset.items()
                }
                aug_obs = aug_dataset['observations']
                print( f'[key {key}] aug_obs count : ', aug_obs.shape)
                for i in range(0, aug_obs.shape[0], process_batch) :
                    aug_obs_batch = aug_obs[i:i+process_batch]
                    batch_len = aug_obs_batch.shape[0]
                    if batch_len != process_batch : # for jitted function
                        aug_obs_batch = np.concatenate([aug_obs_batch, aug_obs_batch[:process_batch-aug_obs_batch.shape[0]]], axis=0)
                    pesudo_actions = self.model.eval_model(aug_obs_batch)[:,0,:]
                    aug_dataset['actions'][i:i+batch_len] = pesudo_actions[:batch_len]
                if aug_dataset_total is None :
                    aug_dataset_total = aug_dataset
                else :
                    for k, v in aug_dataset.items() :
                        aug_dataset_total[k] = np.concatenate([aug_dataset_total[k], v], axis=0)
                
        if aug_dataset_total is not None :
            for k, v in aug_dataset_total.items() :
                dataset[k] = np.concatenate([dataset[k], v], axis=0)

            print(f'[pesudo replay] dataset count : ', dataset['observations'].shape)

        return dataset


from clus.models.peftpool.lorabook import LoRABookTrainState, EWCLoRABookTrainState, DyLoRABookManager
from clus.models.peftpool.lorabook import copy_lorabook_by_source_target

# 1. init with option by __init__ function
# 2. set dropout function for apply rank_mask(trainable mask.) also inference mask
    # but, there is only one trainable, then no dropout.
    # dropout follows random uniform.
class DyLoRABookModel(LoRAPoolModel) :
    '''
    pool mask is selection of LoRABook
        ex) [1,2,10] = [1.,1.,0.,0.,0., 0.,0.,0.,0.,1.]
    '''
    def __init__(
            self,
            meta_init=False,
            **kwargs
        ) -> None:
        self.meta_init = meta_init
        super().__init__(**kwargs)
        print(f'[DyLoRABookModel] init')     
    
    def set_lorapool(self) :
        '''
        set lora pool 
        '''
        print(f"[converting model to lora model ...]")
        # get basis of model
        train_state_orig = self.model.train_state

        # lora model setting 
        self.model.model.apply = lorax.lora(self.model.model.apply)
        self.model.model_eval.apply = lorax.lora(self.model.model_eval.apply)

        def decision_fn(path, param):
            if 'embedding' in path:
                print(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.memory_pool_config.lora_dim
            print(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        self.lora_spec = lorax.simple_spec(train_state_orig.params, decision_fn=decision_fn, tune_vectors=False)
        
        lora_params = init_lora_book(
            train_state_orig.params, 
            self.lora_spec, 
            jax.random.PRNGKey(0),
            book_size=self.memory_pool_config.pool_length*self.memory_pool_config.lora_dim,
        )

        from clus.models.peftpool.peft_optimizer.lorabook_optim import adamw_lorapool
        # if self.lora_optimizer_config is None :
        self.lora_optimizer_config ={
            # 'optimizer_cls' : optax.sgd,
            # 'optimizer_kwargs' : {
            #     'learning_rate' : 5e-3,
            # },
            'optimizer_cls' : adamw_lorapool,
            'optimizer_kwargs' : {
                # 'learning_rate' : 1e-3,
                'learning_rate' : 1e-4,
                'weight_decay' : 1e-4,
            },
        }

        lora_optimizer = self.init_lora_optimizer()

        # EDIT here !!
        self.model.train_state = LoRABookTrainState.create(
            train_mask_rank=jnp.zeros((self.memory_pool_config.pool_length,), dtype=jnp.float32),
            apply_fn=self.model.model.apply, 
            params=lora_params, 
            tx=lora_optimizer,
        )

        ### context id to book mapping model
        # self.book_manager = LoRABookManager(
        #     book_size=self.memory_pool_config.pool_length,
        # ) 

        self.book_manager = DyLoRABookManager(
            book_size=self.memory_pool_config.pool_length * self.memory_pool_config.lora_dim,
            rank_size=self.memory_pool_config.lora_dim,
            init_mode=self.memory_pool_config.meta_init_mode,
            ref_dropout=self.memory_pool_config.ref_dropout,
            train_nested_dropout=self.memory_pool_config.train_nested_dropout,
            eval_process=self.memory_pool_config.eval_process,
        )

    def init_lora_optimizer(self) :
        optimizer_config = self.lora_optimizer_config   
        optimizer = optimizer_config['optimizer_cls'](**optimizer_config['optimizer_kwargs'])
        lora_optimizer = wrap_pool_optimizer( 
            optimizer,
            self.lora_spec,   
            pool_cls=LoRABookWeight,
        )
        return lora_optimizer
    
    #### book manager part #### 
    def book_initialize(self, dataset) :
        meta_init_list = self.book_manager.get_meta_init_list(dataset)
        print("[book initialize] metainit_list : ", meta_init_list)
        for meta_init in meta_init_list : 
            source_pool_idx = meta_init['source']
            target_pool_idx = meta_init['target']
            meta_init_params = copy_lorabook_by_source_target(
                params=self.model.train_state.params,
                source=source_pool_idx,
                target=target_pool_idx,
            )
            self.model.train_state = self.model.train_state.replace(params=meta_init_params)

    def set_pool_mask(self, pool_idx, eval=True) :
        book_mask, grad_mask = self.book_manager.get_bookmask(pool_idx, eval=eval)
        replace_params = set_pool_mask(self.model.train_state.params, book_mask)
        if eval : 
            return replace_params
        else : 
            self.model.train_state = self.model.train_state.replace(
                params=replace_params,
                train_mask_rank=grad_mask,
            )
            return None
    
    def reinit_optimizer(self) :
        lora_optimizer = self.init_lora_optimizer()
        self.model.train_state = LoRABookTrainState.create(
            train_mask_rank=jnp.zeros((self.memory_pool_config.pool_length,), dtype=jnp.float32),
            apply_fn=self.model.model.apply, 
            params=self.model.train_state.params.copy(), 
            tx=lora_optimizer,
        )

    #### main training module #### 
    def train_model(self, dataloader, batch_size=1024, first_epoch=False) :
        '''
        unlike the original model.
        memory pool model requires the full dataset for query-key pre-calculation and custom training loop
        ''' 
        dataloader = self.process_key_selection(dataloader, first_epoch=first_epoch)
        if first_epoch and self.meta_init == True :
            self.book_initialize(dataloader.stacked_data)

        # traininig loop
        total_loss = jnp.array(0)
        b_count = 0
        for b_count, batch_tuple in enumerate(dataloader.get_all_batch(batch_size=batch_size, pool_key='key')) :
            if batch_tuple is None :
                print(f'[batch {b_count}] batch is None :: nothing to learn this phase')
                break
            pool_idx , batch = batch_tuple 
            self.set_pool_mask(pool_idx, eval=False)
            if isinstance(self.model, ConditionalDiffusion) :
                cond = batch['observations']
                x = batch['actions']
                metric = self.model.train_model(x=x,cond=cond)
                loss = metric[1]['train/loss']
            else :
                input_batch = {
                    'inputs' : batch['observations'],
                    'labels' : batch['actions'],
                }
                loss = self.model.train_model(input_batch)
            total_loss += loss
        total_loss /= b_count + 1
        return total_loss   

    #### aggregated Evaluation function ####
    def ag_eval_model(self, obs, prev_action=None) :
        query_dict = {
            'observations' : obs.copy(),
            'prev_actions' : prev_action.copy() if prev_action is not None else None,
        }
        query = self.key_module.process_dataset_query(query_dict)
        if obs.ndim == 3 and obs.shape[1] > 1 :
            obs = obs[:,-1:,:]
        max_indicies , cos_sim = self.key_module.retrieve_key(query)
        if max_indicies.ndim == 0 :
            max_indicies = max_indicies[None]

        all_mask = self.book_manager.book_used.copy()
        # all_mask = np.zeros((self.memory_pool_config.pool_length,), np.float32)
        # all_mask[-1] = 1.
        p = set_pool_mask(self.model.train_state.params, all_mask)
        actions = self.model.eval_model(obs, params=p)

        action_ret = actions
        return action_ret


class EWCDyLoraBookModel(DyLoRABookModel) : 
    def __init__(
            self,
            ewc_lambda=0.1,
            **kwargs
        ) -> None:
        self.ewc_lambda = ewc_lambda
        super().__init__(**kwargs)
        print(f'[DyLoraBookModelEWC] init')     

    def set_lorapool(self) :
        print(f"[converting model to ewc-lora model ...]")
        train_state_orig = self.model.train_state

        # lora model setting 
        self.model.model.apply = lorax.lora(self.model.model.apply)
        self.model.model_eval.apply = lorax.lora(self.model.model_eval.apply)

        def decision_fn(path, param):
            if 'embedding' in path:
                print(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.memory_pool_config.lora_dim
            print(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        self.lora_spec = lorax.simple_spec(train_state_orig.params, decision_fn=decision_fn, tune_vectors=False)
        
        lora_params = init_lora_book(
            train_state_orig.params, 
            self.lora_spec, 
            jax.random.PRNGKey(0),
            book_size=self.memory_pool_config.pool_length*self.memory_pool_config.lora_dim,
        )

        from clus.models.peftpool.peft_optimizer.lorabook_optim import adamw_lorapool
        self.lora_optimizer_config ={
            'optimizer_cls' : adamw_lorapool,
            'optimizer_kwargs' : {
                # 'learning_rate' : 1e-3,
                'learning_rate' : 1e-4,
                'weight_decay' : 1e-4,
            },
        }
        lora_optimizer = self.init_lora_optimizer()

        def lorabook_copy(lora_params, zeros_params=False) :
            '''
            only copy the lora pool params for efficiency 
            '''
            def _lorabook_copy(path, p) :
                name = path[-1]
                if name == 'a' or name == 'b' : # (book,N)
                    if zeros_params :
                        return jnp.zeros_like(p)
                    else :
                        return jnp.array(p, copy=True)
                else :
                    return None
            return tree_util.tree_map(_lorabook_copy, lora_params)
        
        self.model.train_state = EWCLoRABookTrainState.create(
            train_mask_rank=jnp.zeros((self.memory_pool_config.pool_length,), dtype=jnp.float32),
            apply_fn=self.model.model.apply, 
            params=lora_params, 
            ewc_params=lorabook_copy(lora_params, zeros_params=False),
            fisher_params=lorabook_copy(lora_params, zeros_params=True),
            tx=lora_optimizer,
        )

        self.book_manager = DyLoRABookManager(
            book_size=self.memory_pool_config.pool_length * self.memory_pool_config.lora_dim,
            rank_size=self.memory_pool_config.lora_dim,
            init_mode=self.memory_pool_config.meta_init_mode,
            ref_dropout=self.memory_pool_config.ref_dropout,
            train_nested_dropout=self.memory_pool_config.train_nested_dropout,
            eval_process=self.memory_pool_config.eval_process,
        )

    # EWC loss term a
    def update_loss_fn(self) :
        '''
        set every loss_fn to get default input configures
            (params, state, inputs)
        '''
        self.ewc_ratio = 1.
        self.ewc_mode = 'ewc' # or l2

        if self.ewc_mode == 'ewc' :
            reg_fn = lambda x, y, z: jnp.sum(z * jnp.square(x - y))
        elif self.ewc_mode == 'l2' :
            reg_fn = lambda x, y, z: jnp.sum(jnp.square(x - y))
        
        # check code entangled for novel update
        base_fn = self.model.loss_fn 
        
        def calc_ewc_loss(params, state, **kwargs) :
            '''
            input argument is same as original model(conditional diffusion or others)
            '''
            ewc_loss = tree_util.tree_map(
                reg_fn,
                state.params, 
                state.ewc_params,
                state.fisher_params,
            )
            return ewc_loss

        def ewc_loss_wrapper(**kwargs) :
            loss = base_fn(**kwargs)
            ewc_loss = calc_ewc_loss(**kwargs)
            return loss + self.ewc_lambda * ewc_loss
        
        self.model.loss_fn = ewc_loss_wrapper
        

    def train_model(self, dataloader, batch_size=1024, first_epoch=False) :
        '''
        unlike the original model.
        memory pool model requires the full dataset for query-key pre-calculation and custom training loop
        ''' 
        dataloader = self.process_key_selection(dataloader, first_epoch=first_epoch)
        if first_epoch :
            self.prev_params = jax.tree_map(lambda x: jnp.array(x, copy=True), self.model.train_state.params)
            if self.meta_init == True :
                self.book_initialize(dataloader.stacked_data)

        # traininig loop
        total_loss = jnp.array(0)
        b_count = 0
        for b_count, batch_tuple in enumerate(dataloader.get_all_batch(batch_size=batch_size, pool_key='key')) :
            if batch_tuple is None :
                print(f'[batch {b_count}] batch is None :: nothing to learn this phase')
                break
            pool_idx , batch = batch_tuple 
            self.set_pool_mask(pool_idx, eval=False)
            if isinstance(self.model, ConditionalDiffusion) :
                cond = batch['observations']
                x = batch['actions']
                metric = self.model.train_model(x=x,cond=cond)
                loss = metric[1]['train/loss']
            else :
                input_batch = {
                    'inputs' : batch['observations'],
                    'labels' : batch['actions'],
                }
                loss = self.model.train_model(input_batch)
            total_loss += loss

class DyLoRABookModelOracle(DyLoRABookModel) :
    def eval_model(self, obs, prev_action=None, return_unique=False, daco_query=None) : # multi evaluation function (core!)
        # obs is set to be (B, S, F)
        # Must DACo query (B, S, F) comes from Evaluator.
        query = daco_query.copy()

        if obs.ndim == 3 and obs.shape[1] > 1 :
            obs = obs[:,-1:,:]
        max_indicies , cos_sim = self.key_module.retrieve_key(query)
        if max_indicies.ndim == 0 :
            max_indicies = max_indicies[None]
        unique_indices = np.unique(max_indicies) 
        for pool_idx in unique_indices :
            p = self.set_pool_mask(pool_idx, eval=True)
            action = self.model.eval_model(obs, params=p)
            indicies_mask = (max_indicies == pool_idx).astype(int)[:,None]
            action = action * indicies_mask
            if pool_idx == unique_indices[0] :
                action_ret = action
            else :
                action_ret += action
        if return_unique :
            return action_ret, unique_indices
        return action_ret

from jax import tree_util

DEFAULT_VARIENT_CONFIGS = [
    {
        'key_name': 'skill',
        'num_keys': 10,
        'feature_dim': 512,
        'query_method': 'mean'
    },
    {
        'key_name': 'observations',
        'num_keys': 10,
        'feature_dim': 140,
        'query_method': 'mean'
    },
]
DEFAULT_ALPHAS = jnp.array([1, 11])



from dataclasses import dataclass, field
from contextlib import contextmanager

def mask_gradients(grads, pattern):
    """
    Mask gradients to keep only those matching the pattern.
    for qax implicitarray object
    pattern is a list of strings.
    """
    def mask_fn(path, g):
        name = path[-1]
        if name in pattern : 
            print(path)
            return g
        else :
            return jnp.zeros_like(g)
    return jax.tree_util.tree_map_with_path(mask_fn, grads)

class DualTrainState(train_state.TrainState) :
    mask_list: list[str] = field(default_factory=lambda: [None])
    def apply_gradients(self, *, grads, **kwargs):
        masked_grads = mask_gradients(grads, self.mask_list) # 1103 beware NOTE in adam like optimzer
        updates, new_opt_state = self.tx.update(masked_grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

class DualLoRAPoolModel(LoRAPoolModel) :
    '''
    wrapper function for DualLoRAPool training 
    one-for generalization other for task specific
    '''
    def set_lorapool(self) :
        '''
        set lora pool 
        '''
        print(f"[converting model to DUAL_lora model ...]")
        # get basis of model
        train_state_orig = self.model.train_state

        # lora model setting 
        self.model_apply = lorax.lora(self.model.model.apply)
        self.model.model_eval.apply = lorax.lora(self.model.model_eval.apply)

        def decision_fn(path, param):
            if 'embedding' in path:
                print(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.memory_pool_config.lora_dim
            print(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        self.lora_spec = lorax.simple_spec(train_state_orig.params, decision_fn=decision_fn, tune_vectors=True)
        
        lora_params = init_lora_pool(
            train_state_orig.params, 
            self.lora_spec, 
            jax.random.PRNGKey(0),
            pool_size=self.memory_pool_config.pool_length,
            mode='dual',
        )

        self.lora_optimizer_config ={
            'optimizer_cls' : optax.adamw,
            'optimizer_kwargs' : {
                'learning_rate' : 1e-4,
                'weight_decay' : 1e-4,
            },
        }

        lora_optimizer = self.init_lora_optimizer()
        # create train state by lora params
        self.model.train_state = DualTrainState.create(
            apply_fn=self.model_apply, 
            params=lora_params, 
            tx=lora_optimizer,
        )

    def init_lora_optimizer(self) :
        optimizer_config = self.lora_optimizer_config   
        optimizer = optimizer_config['optimizer_cls'](**optimizer_config['optimizer_kwargs'])
        lora_optimizer = wrap_pool_optimizer( 
            optimizer,
            self.lora_spec,   
            mode='dual',
        )
        return lora_optimizer
    
    #### main training module #### 
    def train_model(self, dataloader, batch_size=1024) :
        '''
        unlike the original model.
        memory pool model requires the full dataset for query-key pre-calculation and custom training loop
        ''' 
        if not isinstance(dataloader, MemoryPoolDataloader) :
            raise Exception(f"[ERROR] dataloader type must be MemoryPoolDataloader in LoRAModel \nbut, {type(dataloader)}")
        
        # calculate the query for dataloader and save them in dataloder.stacked_data['query'] 
        # TODO NOTE if you want to change the strategy for query calculation, you should change this part 
        if 'query' not in dataloader.stacked_data.keys() :
            dataloader.stacked_data['query'] = self.query_calculation(dataloader.stacked_data['observations'])

        # calculate key of memory-pool and query similarity
        max_prob_indices_list = []
        n_queries = dataloader.stacked_data['query'].shape[0]

        for i in range(0, n_queries, batch_size):
            query_batch = dataloader.stacked_data['query'][i:i+batch_size]  # Shape will be (1024, F)
            prob = self.get_key_prob_jit(self.key_module_params, query_batch)  # Assuming that prob.shape is (1024, B)
            # balancing algo 
            balanced_prob = prob * self.key_appear_prob_reverse
            max_prob_indices = jnp.argmax(balanced_prob, axis=-1)
            max_prob_indices_list.append(max_prob_indices)

        max_prob_indices_list = jnp.concatenate(max_prob_indices_list, axis=0)

        # balancing algo 
        counts = np.bincount(max_prob_indices_list, minlength=self.memory_pool_config.pool_length)
        self.key_appear_count += jnp.array(counts, dtype=jnp.float32) / jnp.sum(counts)

        dataloader.stacked_data['key'] = max_prob_indices_list

        # update the key of memory pool by query-key similarity
        self.key_module_train_state, metric = self.train_key_model_jit(
            self.key_module_train_state, 
            dataloader.stacked_data['query'],
            self.key_appear_prob_reverse,    
        )

        ## traininig loop general
        self.model.train_state.replace(mask_list=['a','b'])
        pool_mask = np.zeros((self.memory_pool_config.pool_length,), jnp.float32)
        self.model.train_state.replace(params=set_pool_mask(self.model.train_state.params, pool_mask))
        for b_count, batch in enumerate(dataloader.get_all_batch(batch_size=batch_size, pool_key=None)) :
            if isinstance(self.model, ConditionalDiffusion) :
                cond = batch['observations']
                x = batch['actions']
                metric = self.model.train_model(x=x,cond=cond)
                loss = metric[1]['train/loss']
            else :
                input_batch = {
                    'inputs' : batch['observations'],
                    'labels' : batch['actions'],
                }
                loss = self.model.train_model(input_batch)

        ## traininig loop task_specific
        total_loss = 0
        self.model.train_state.replace(mask_list=['a_g','b_g'])
        for b_count, batch_tuple in enumerate(dataloader.get_all_batch(batch_size=batch_size, pool_key='key')) :
            pool_idx , batch = batch_tuple  
            # set model mask
            pool_mask = np.zeros((self.memory_pool_config.pool_length,), jnp.float32)
            pool_mask[pool_idx] = 1. # NOTE : pool_idx is list of index
            self.model.train_state.replace(params=set_pool_mask(self.model.train_state.params, pool_mask))

            if isinstance(self.model, ConditionalDiffusion) :
                cond = batch['observations']
                x = batch['actions']
                metric = self.model.train_model(x=x,cond=cond)
                loss = metric[1]['train/loss']
            else :
                input_batch = {
                    'inputs' : batch['observations'],
                    'labels' : batch['actions'],
                }
                loss = self.model.train_model(input_batch)
            total_loss += loss
        total_loss /= b_count
        return total_loss   
    
    def reinit_optimizer(self) :
        lora_optimizer = self.init_lora_optimizer()
        self.model.train_state = DualTrainState.create(
            apply_fn=self.model_apply, 
            params=self.model.train_state.params.copy() , 
            tx=lora_optimizer,
        )

    @contextmanager
    def use_general_optimizer(self):
        self.model.train_state, self.model.train_state_g = self.model.train_state_g, self.model.train_state
        try:
            yield  
        finally:
            # sync params 
            self.model.train_state_g.replace(params=self.model.train_state.params)
            self.model.train_state, self.model.train_state_g = self.model.train_state_g, self.model.train_state

import cloudpickle  
if __name__ == "__main__" : 

    # example code for multi-query query-key similarity
    variant_configs = [
            {
                'key_name': 'skill',
                'num_keys': 10,
                'feature_dim': 512,
            },
            {
                'key_name': 'traj',
                'num_keys': 10,
                'feature_dim': 140,
            },
        ]
    
    scale = jnp.array([1., 1.])

