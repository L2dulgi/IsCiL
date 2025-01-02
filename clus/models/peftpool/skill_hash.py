from clus.models.peftpool.pool_basis import *
from clus.env.offline import MemoryPoolDataloader
from tqdm import tqdm
import jax
import optax
import numpy as np

from flax.training import train_state
from functools import partial

from clus.models.peftpool.pool_basis import LoRAMemoryPoolConfig
from sklearn.cluster import KMeans

DEFAULT_OPTIMIZER_CONFIG = {
    'optim_cls' : optax.adamw,
    'optim_kwargs' : {'learning_rate': 1e-4, 'weight_decay': 1e-4}
}
# skill hash == adapter matching module
# BaseHashModel
# - ContextHashModel
#   - multi key is supported by params
# - DualHashModel
#   - MultiKeyDualHashModel

class BaseHashModel() :
    def __init__(self) :
        pass

    ## core function ## 
    def process_dataset_key(self, dataset) :
        '''
        dataset key processing further processing for dataset
        '''
        raise NotImplementedError
    
    def retrieve_key(self, query) :
        '''
        single retrieval function for key 
        '''
        raise NotImplementedError
    
    def process_dataset_query(self, dataset) :
        '''
        query method for dataset
        '''
        raise NotImplementedError
    
    ## key updating function ## 
    def update_model(self, dataset) :
        ''' 
        by jit or something update function
        highly related to the retrieve_key function.
        '''
        raise NotImplementedError
    
    def get_seen_context_idx(self) :
        '''
        replay augmentation for dataset
        beware, this function only called once in the each phase
        '''
        raise NotImplementedError
    # 1128 pesudo replay function added

import pickle
class ContextHashModel(BaseHashModel) :
    def __init__(
            self,
            memory_pool_config = LoRAMemoryPoolConfig(
                pool_length=30,
                feature_dim=652,
                lora_dim=32,
                embedding_key="split",
            ),
            key_mode = 'single', # single,multi,CoLoR,DACo,TAIL
        ):
        self.memory_pool_config = memory_pool_config
        self.env_name = 'mm' if self.memory_pool_config.feature_dim == 652 else 'kitchen'
        if self.memory_pool_config.embedding_key == 'base' :
            self.memory_pool_config.feature_dim = self.memory_pool_config.feature_dim - 512
        if self.memory_pool_config.embedding_key == 'tailg' :
            self.memory_pool_config.feature_dim = 512

        print(f"[Context Hash Model] key_mode : {key_mode}")
        self.key_mode = key_mode
        # below two is used in CoLoR
        self.key_num = memory_pool_config.key_num
        self.key_idx = 0
        if self.key_mode == 'single' or self.key_mode == 'DACo' or self.key_mode == 'tailg':
            self.key_module = CosineSimilarityModule(
                num_keys=self.memory_pool_config.pool_length, 
                feature_dim=self.memory_pool_config.feature_dim
            )
        elif self.key_mode == 'multi' or self.key_mode == 'CoLoR' :
            self.key_module = MultiKeySimilarityModule(
                num_pool=self.memory_pool_config.pool_length, 
                key_num=self.key_num,
                feature_dim=self.memory_pool_config.feature_dim
            )
        else :
            raise NotImplementedError

        key_module_params = self.key_module.init(
            jax.random.PRNGKey(0), 
            jnp.zeros((1, self.memory_pool_config.feature_dim))
        )
        if self.key_mode == 'CoLoR' or self.key_mode == 'TAIL':
            key_module_params['params']['keys'] = key_module_params['params']['keys'].at[:].set(1.)
        elif self.key_mode == 'tailg' :
            if self.env_name == 'mm' :
                print('[MMworld key initialization]')
                with open('data/continual_dataset/evolving_world/mm_lang_embedding.pkl', 'rb') as f :
                    data = pickle.load(f)
                    stack = jnp.stack(list(data.values()))
                    li = stack.shape[0]
                    key_module_params['params']['keys'] = key_module_params['params']['keys'].at[:li].set(stack)

            elif self.env_name == 'kitchen':
                print('[Kitchen key initialization]')
                with open('data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl', 'rb') as f :
                    data = pickle.load(f)
                    stack = jnp.stack(list(data.values()))
                    li = stack.shape[0]
                    key_module_params['params']['keys'] = key_module_params['params']['keys'].at[:li].set(stack)

        self.key_module_train_state = train_state.TrainState.create(
            apply_fn=self.key_module.apply, 
            params=key_module_params, 
            tx=optax.adamw(learning_rate=1e-4, weight_decay=1e-4),
        )

        self.key_appear_count = np.ones((self.memory_pool_config.pool_length,), jnp.float32)
        self.key_appear_count_fixed = self.key_appear_count.copy()
    
    '''
    daco TODO 
    init datset using daco_context in dataloading 
    init all dacocontext in advance - by pkl file - dualL2MOracle 
        daco_query processing in the dual l2m
    '''
    ## core function ## 
    def process_dataset_key(self, dataset, batch_size=1024, task_id=None) :
        if self.key_mode == 'single' or self.key_mode == 'multi' or self.key_mode == 'DACo' :
            # retrieve the key for L2M #
            max_prob_indices_list = []
            prob_list = []
            n_queries = len(dataset['query'])
            for i in range(0, n_queries, batch_size):
                query_batch = dataset['query'][i:i+batch_size] 
                prob = self.get_key_prob_jit(self.key_module_train_state, query_batch)  
                balanced_prob = prob * self.key_appear_prob_reverse
                max_prob_indices = np.argmax(balanced_prob, axis=-1)
                max_prob_indices_list.append(max_prob_indices)
                prob_list.append(prob)
            max_prob_indices_list = np.concatenate(max_prob_indices_list, axis=0)
            prob_list = np.concatenate(prob_list, axis=0)

            dataset['key'] = max_prob_indices_list
            dataset['probs'] = prob_list
            counts = np.bincount(max_prob_indices_list, minlength=self.memory_pool_config.pool_length)
            self.key_appear_count += np.array(counts, dtype=jnp.float32) / np.sum(counts)
        
        elif self.key_mode == 'tailg' :
            # retrieve the key for L2M #
            max_prob_indices_list = []
            prob_list = []
            n_queries = len(dataset['query'])
            for i in range(0, n_queries, batch_size):
                query_batch = dataset['query'][i:i+batch_size] 
                prob = self.get_key_prob_jit(self.key_module_train_state, query_batch)  
                max_prob_indices = np.argmax(prob, axis=-1)
                max_prob_indices_list.append(max_prob_indices)
                prob_list.append(prob)
            max_prob_indices_list = np.concatenate(max_prob_indices_list, axis=0)
            prob_list = np.concatenate(prob_list, axis=0)

            dataset['key'] = max_prob_indices_list
            dataset['probs'] = prob_list
            counts = np.bincount(max_prob_indices_list, minlength=self.memory_pool_config.pool_length)
            self.key_appear_count += np.array(counts, dtype=jnp.float32) / np.sum(counts)

        elif self.key_mode == 'CoLoR' :
            # initialize the fixed key of CoLoR #
            context_mean = KMeans(n_clusters=self.key_num, n_init=10).fit(dataset['query']).cluster_centers_
            normalized_context_anchor = context_mean / np.linalg.norm(context_mean, axis=-1, keepdims=True)    
            params = self.key_module_train_state.params
            params['params']['keys'] = params['params']['keys'].at[self.key_idx].set(normalized_context_anchor)
            self.key_module_train_state = self.key_module_train_state.replace(params=params)
            dataset['key'] = np.zeros((len(dataset['query']),), dtype=np.int32)
            dataset['key'][:] = self.key_idx 
            self.key_appear_count[self.key_idx] += 1
            self.key_idx += 1
        elif self.key_mode == 'DACoFix' : # works! 
            print(f"key mode : {self.key_mode}, with context {len(np.unique(dataset['daco_context']))}")
            dataset['key'] = np.zeros((len(dataset['query']),), dtype=np.int32)
            for i, context in enumerate(np.unique(dataset['daco_context'])) :
                print(f"key mode : {self.key_mode}, with context")
                normalized_context_anchor = context.copy() / np.linalg.norm(context)
                params = self.key_module_train_state.params
                params['params']['keys'] = params['params']['keys'].at[self.key_idx].set(normalized_context_anchor)
                indicies = np.where(dataset['daco_context'] == context)[0]
                dataset['key'][indicies] = self.key_idx
                self.key_appear_count[self.key_idx] += 1
                self.key_idx += 1 
        else : 
            raise NotImplementedError
        return dataset
        
    def retrieve_key(self, query, task_id=None) :
        if type(query) == dict :
            query = query['query'] 
        # if daco_query is not None :
        #     query = daco_query[...,:30]
        if self.memory_pool_config.embedding_key == 'base' :
            query = query[:,:,:self.memory_pool_config.feature_dim]
        elif self.memory_pool_config.embedding_key == 'tailg' :
            query = query[:,:,-512:]
        prob = self.get_key_prob_jit(self.key_module_train_state, query)
        key_idx = jnp.argmax(prob, axis=-1)
        return key_idx, prob
    
    def process_dataset_query(self, dataset, daco_query=None) :
        # data : (b,F)
        query_method = self.memory_pool_config.embedding_key
        if isinstance(dataset, dict) :
            data = dataset['observations'].copy()
            if self.key_mode == 'DACo' :
                dataset['query'] = dataset['daco_context'].copy()
            elif query_method == "split" or query_method == "base":
                dataset['query'] = data[:,:self.memory_pool_config.feature_dim] 
            elif query_method == "tailg" :
                dataset['query'] = data[:,-512:]
            else:
                print("Invalid query method.")
                raise NotImplementedError
            return dataset
        else :
            data = dataset.copy()
            if self.key_mode == 'DACo' :
                data = daco_query.copy()
            elif query_method == "split" or query_method == "base":
                data = data[:,:self.memory_pool_config.feature_dim] 
            elif query_method == "tailg" :
                data = data[:,-512:]
            else:
                print("Invalid query method.")
                raise NotImplementedError
            return data
    
    ## Aux function (only used for modulator ensembel)##
    def retrieve_aux_key(self, query) :
        if type(query) == dict :
            query = query['query']  
        prob = self.get_key_prob_jit(self.key_module_train_state, query)
        key_idx = jnp.argmax(prob, axis=-1).squeeze()
        mask = np.zeros_like(prob)
        mask[np.arange(mask.shape[0]),:, key_idx] = 1.0
        print(key_idx)

        max_prob = prob[np.arange(prob.shape[0]),:, key_idx] - 0.0005
        second_prob = prob.at[np.arange(prob.shape[0]),:, key_idx].set(-jnp.inf)
        second_key_idx = jnp.argmax(second_prob[:, :, :10], axis=-1).squeeze()
        # mask[np.arange(mask.shape[0]),:, second_key_idx] = 1.0

        mask[np.arange(mask.shape[0]),:, second_key_idx] = second_prob[np.arange(mask.shape[0]),:, second_key_idx] > max_prob

        print(mask[np.arange(mask.shape[0]),:, second_key_idx].squeeze() )

        # mask[np.arange(mask.shape[0]),:, second_key_idx] = prob[np.arange(mask.shape[0]),:, second_key_idx].squeeze()

        return mask, prob

    ## key updating function ## TAIL [v]
    def update_model(self, dataset, batch_size=1024) :
        if self.key_mode == 'CoLoR' or self.key_mode == 'tailg' : # no update for multikey(CoLoR)
            return 0
        n_queries = len(dataset['query'])
        for i in range(0, n_queries, batch_size):
            query_batch = dataset['query'][i:i+batch_size] 
            self.key_module_train_state, metric = self.train_key_model_jit(
                self.key_module_train_state, 
                query_batch,
                self.key_appear_prob_reverse,    
            )
        return metric
    
    ## jitted key function ## 
    @partial(jax.jit, static_argnums=(0,))
    def get_key_prob_jit(self, state, query) :
        return state.apply_fn(state.params, query)  
    
    @partial(jax.jit, static_argnums=(0,))
    def train_key_model_jit(self, state, query, balancing_prob, rngs=None):
        grad_fn = jax.grad(max_cosine_similarity_loss, has_aux=True)
        grads, metric = grad_fn(state.params, state, query, balancing_prob, rngs=rngs)
        state = state.apply_gradients(grads=grads)
        return state, metric

    ## utill function ##
    @property   
    def key_appear_prob_reverse(self) : # Temporal
        # return np.ones((self.memory_pool_config.pool_length,), dtype=np.float32)
        # key_appear = self.key_appear_count if self.first_phase else self.key_appear_count_fixed 
        key_appear = self.key_appear_count_fixed 
        min_key_appear = np.min(key_appear)
        max_key_appear = np.max(key_appear)
        range_key_appear = max_key_appear - min_key_appear
        denominator = np.maximum(range_key_appear, 1)

        pre_softmax_prob = 1 - (key_appear - min_key_appear) / denominator
        return pre_softmax_prob 

class DualHashModel(BaseHashModel):
    ## initialization ##
    def __init__(
            self, 
            skill_dim=512,
            context_dim=140,
            sg_optimizer_config=DEFAULT_OPTIMIZER_CONFIG,
            cm_optimizer_config=DEFAULT_OPTIMIZER_CONFIG,
            memory_pool_config = LoRAMemoryPoolConfig( 
                    pool_length=50,
                    feature_dim=652,
                    lora_dim=32,
                    embedding_key='split',
            ),
            context_mode='obs', # obs, traj
            balance_context_prob=False,
            retrieval_mode='nsim', # sim(similarity) , nsim(normalized similarity)
            force_fixed=False, # default False force the novel context do not inturrupt
            learnable_key=True,
        ): 
        self.memory_pool_config = memory_pool_config
        self.balance_context_prob = balance_context_prob
        print(f"balance_context_prob in DualHashmodel : {self.balance_context_prob}")

        ## key dimension selection ## 
        self.num_keys = self.memory_pool_config.pool_length
        self.skill_dim = skill_dim
        self.orig_context_dim = context_dim
        self.context_dim = context_dim - 512
    
        self.context_mode = context_mode
        self.retrieval_mode = retrieval_mode
        self.force_fixed = force_fixed
        self.learnable_key = learnable_key

        if self.context_dim == 60 : 
            print("kitchen context dim 60 to 30")
            self.context_dim -= 30
        if self.context_mode == 'dyna' :
            self.context_dim += 9 # TODO automate this by sample input of model


        if self.orig_context_dim == 39 :
            print("[skill hash model]continual world setting")
            self.context_mode = 'cw10'
            self.context_dim = self.orig_context_dim



        # Initialize optimizers with provided configurations
        self.sg_optimizer = sg_optimizer_config['optim_cls'](**sg_optimizer_config['optim_kwargs'])
        self.cm_optimizer = cm_optimizer_config['optim_cls'](**cm_optimizer_config['optim_kwargs'])
        self.init_skill_group_module()
        self.init_context_match_module()

        self.key_appear_count = np.ones((self.memory_pool_config.pool_length,), jnp.float32)
        self.key_appear_count_fixed = self.key_appear_count.copy()
  
    def init_skill_group_module(self) :
        self.skill_group_module = CosineSimilarityModule(
            num_keys=self.num_keys, 
            feature_dim=self.skill_dim, 
        )
        skill_group_module_params = self.skill_group_module.init(
            jax.random.PRNGKey(0), 
            jnp.zeros((1, self.skill_dim))
        )
        self.skill_group_module_train_state = train_state.TrainState.create(
            apply_fn=self.skill_group_module.apply, 
            params=skill_group_module_params, 
            tx=self.sg_optimizer,
        )

        self.skill_group = [[] for _ in range(self.num_keys)] 
        # skill_counts is used to determine whether the skill is novel or not
        self.skill_counts = np.zeros((self.num_keys,), jnp.float32)
        # skill_threshold is used to determine whether the skill is novel or not
        self.skill_threshold = np.ones((self.num_keys,), jnp.float32)
        self.context_threshold = np.ones((self.num_keys,), jnp.float32)
        self.skill_group_len = 0
        self.used_context_len = 0
        self.used_context_len_prev = 0

    def init_context_match_module(self) :
        self.context_module = CosineSimilarityModule(
            num_keys=self.num_keys, 
            feature_dim=self.context_dim, 
        )
        context_module_params = self.context_module.init(
            jax.random.PRNGKey(0), 
            jnp.zeros((1, self.context_dim))
        )
        self.context_module_train_state = train_state.TrainState.create(
            apply_fn=self.context_module.apply, 
            params=context_module_params, 
            tx=self.cm_optimizer,
        )
        
    ## core function ## 
    def process_dataset_key(self, dataset) :
        '''
        process_skill_context
        Process the training dataset.
        Args:
            dataset : The input dataset to be processed.
            needed the ['skill_query', 'context_query'] appended
        Returns:
            tuple: A tuple containing context_matched_dataset (list) and module_reinit_dict (dict).
                dataset : ['skill_query', 'context_query', 'skill_group_idx', 'context_idx'] appended
        '''
        if 'skill_query' not in dataset.keys() :
            raise ValueError(f'[skill_query] not in dataset keys {dataset.keys()}')
        if 'context_query' not in dataset.keys() :
            raise ValueError(f'[context_query] not in dataset keys {dataset.keys()}')
        
        module_reinit_dict = {}

        self.used_context_len_prev = self.used_context_len
        # skill group matched. dataset['skill_group_idx', 'context_idx] is appended
        matched_dataset, unique_dict = self.dataset_skill_groupping(dataset)
        matched_dataset, module_reinit_dict = self.find_context_matching(matched_dataset)        
        matched_dataset['key'] = matched_dataset['context_idx']
        matched_dataset['probs'] = matched_dataset['context_prob']

        if self.force_fixed : # forcing the novel skill exist, then utilize to learn the novel context forced.
            for novel_context in unique_dict['ss_nc'] :
                # replace the context idx is 0 to the novel context idx
                matched_dataset['key'][matched_dataset['skill_group_idx'] == novel_context[0]] = novel_context[1]

        counts = np.bincount(matched_dataset['key'], minlength=self.memory_pool_config.pool_length)
        self.key_appear_count += np.array(counts, dtype=jnp.float32) / np.sum(counts)

        # dataset attachable retrun value
        return matched_dataset, module_reinit_dict
    
    def retrieve_key(self, query) :
        '''
        Evaluate the current query.
        Args:
            query(dictionary): (skill_query, context_query) pair.
        Returns:
            int: The index of the context.
        '''
        skill_embedding = query['skill_query']
        context_query = query['context_query']

        skill_group_idx, sprob = self.retrieve_skill_group(skill_embedding)
        context_idx, cprob = self.retrieve_context(
            context_query=context_query,
            skill_group_idx=skill_group_idx,
        )
        # for i, se in enumerate(skill_group_idx ):
        #     print(i, se, context_idx[i])
        #     print(self.skill_group, se)
        #     if context_idx[i] not in  self.skill_group[se[0]] :
        #         print(f"skill {se} is not in the skill group {self.skill_group[int(se)]}")  
        #         raise ValueError
        return context_idx, cprob
    
    def process_dataset_query(self, dataset, eval_flag=False) :
        self.traj_len = 10
        traj_len = self.traj_len
        if self.context_mode == 'obs' :
            if type(dataset) == dict :
                dataset['skill_query'] = dataset['observations'][..., -512:]
                dataset['context_query'] = dataset['observations'][..., :self.context_dim]
                # Temporally context_query
                # dataset['context_query'] = dataset['context_query'][:,:30]
                return dataset
            else : # for normal object
                raise NotImplementedError
                # data = dataset
                # return data[...,-512:], data[...,:self.context_dim]
        elif self.context_mode == 'traj' :
            if type(dataset) == dict :
                dataset['skill_query'] = dataset['observations'][:, -512:].copy()
                dataset['context_query'] = dataset['observations'][:, :self.context_dim].copy()
                for j in range(len(dataset['context_query'])) :
                    aggregate_start = max(j-traj_len,dataset['episode_boundary'][j][0])
                    aggregate_traj = dataset['context_query'][aggregate_start:j+1,:].copy()
                    if aggregate_traj.shape[0] < traj_len :
                        # extend the first element
                        traj_padding = np.tile(aggregate_traj[:1, :] ,(traj_len - aggregate_traj.shape[0], 1))
                        aggregate_traj = np.concatenate([traj_padding, aggregate_traj], axis=0)
                    aggregated_context = np.mean(aggregate_traj, axis=0)
                    dataset['context_query'][j] = aggregated_context
                return dataset
            else :
                # check if dataset if trajectory (B, S, F)
                data = dataset
                skill_context = data[:,-1:,-512:]
                traj_context = np.mean(data[:,:,:self.context_dim], axis=1, keepdims=True)
                return skill_context, traj_context
        elif self.context_mode == 'dyna' :
            if type(dataset) == dict :
                    dataset['skill_query'] = dataset['observations'][:, -512:]
                    dataset['context_query'] = dataset['observations'][:, :self.context_dim]
                    if 'prev_actions' not in dataset.keys() :
                        dataset['prev_actions'] = dataset['actions'][:] # NOTE TODO fill this place if dyna is used for clustering
                    dataset['context_query'] = np.concatenate([dataset['context_query'], dataset['prev_actions']], axis=-1)
                    return dataset
            else : 
                raise f"invalid dataset type {type(dataset)}" 
        elif self.context_mode == 'cw10' :
            if type(dataset) == dict :
                dataset['skill_query'] = np.ones(dataset['observations'].shape[:-1]+(512,), dtype=np.float32)
                dataset['context_query'] = dataset['observations'][..., :self.context_dim]
                return dataset
            else : # for normal object
                raise NotImplementedError
        else :
            raise NotImplementedError
            
    ## update_model ##
    def update_model(self, dataset, batch_size=1024) :
        '''
        Update the model with the current dataset. with heuristic
        Args:
            dataset : The input dataset to be processed.
        '''
        if self.learnable_key == True :
            n_queries = len(dataset['context_query'])
            for i in range(0, n_queries, batch_size):
                query_batch = dataset['context_query'][i:i+batch_size] 
                self.context_module_train_state, metric = self.train_context_model_jit(
                    self.context_module_train_state, 
                    query_batch,
                    self.key_appear_prob_reverse,    
                )
            return metric
        else : 
            return 0

    @partial(jax.jit, static_argnums=(0,))
    def train_context_model_jit(self, state, query, balancing_prob, rngs=None):
        grad_fn = jax.grad(max_cosine_similarity_loss, has_aux=True)
        grads, metric = grad_fn(state.params, state, query, balancing_prob, rngs=rngs)
        state = state.apply_gradients(grads=grads)
        return state, metric

    def get_seen_context_idx(self) :
        '''
        replay augmentation for dataset
        beware, this function only called once in the each phase
        '''
        seen_context = []
        for i in range(self.used_context_len_prev) :
            seen_context.append(i)
        return seen_context
    
    ## phase skill grouping processors ##
    def dataset_skill_groupping(self, dataset) :
        # known/unknown semantic skill processing
        print('update_skill_group', dataset['skill_query'].shape)
        unique_dict = self.update_skill_group(dataset)
        skill_group_indicies = self.find_skill_group(dataset)
        dataset['skill_group_idx'] = skill_group_indicies
        print("[skill group]", self.skill_group[:self.skill_group_len] )
        return dataset, unique_dict

    def create_skill_group(
            self, 
            skill_group_idx,
            context_idx,
            skill_query, 
            dataset,
        ) :
        params = self.skill_group_module_train_state.params
        normalized_skill = skill_query / np.linalg.norm(skill_query)
        params['params']['keys'] = params['params']['keys'].at[skill_group_idx].set(normalized_skill)
        self.skill_group_module_train_state = self.skill_group_module_train_state.replace(params=params)  
    
        self.skill_threshold[skill_group_idx] = 0.99 # set key to 1.0 similarity vector

        self.expand_skill_group(    
            skill_group_idx=skill_group_idx,
            context_idx=context_idx,
            skill_query=skill_query,
            dataset=dataset,
        )
        
        self.skill_group_len += 1

    def expand_skill_group(
            self,
            skill_group_idx,
            context_idx,
            skill_query,
            dataset,
        ) : 
        if self.used_context_len >= self.num_keys :
            print(f"context is full! novel context is not added existing context will be selected.")
            return

        # get context indicies from dataset (maximum length k)
        skill_matched_idx = np.where(np.all(dataset['skill_query'] == skill_query, axis=1))[0]
        cq = dataset['context_query'][skill_matched_idx]
        context_mean = np.mean(cq, axis=0)

        # context_indicies = self.init_context_match(dataset, skill) 
        context_indicies = [context_idx] 
        params = self.context_module_train_state.params
        normalized_context_anchor = context_mean / np.linalg.norm(context_mean)    
        normalized_context = cq / np.linalg.norm(cq, axis=1, keepdims=True)

        # # context thresholding mehtod (median similarity @ 1115) => should check the update function.
        context_sim_min = np.median(normalized_context @ normalized_context_anchor.T)
        self.context_threshold[context_idx] = context_sim_min

        params['params']['keys'] = params['params']['keys'].at[context_idx].set(normalized_context_anchor)
        self.context_module_train_state = self.context_module_train_state.replace(params=params)
        self.skill_group[skill_group_idx].extend(context_indicies)

        self.used_context_len += 1

    ## phase dataset processors ##
    def update_skill_group(self, dataset) :
        '''
        Update the skill group by context matching(Novel semantic)
            seen_skill
                seen_context -> do nothing
                novel_context -> expand skill group
            novel_skill
                seen_context -> existing skill 
                novel_context -> novel skill group generation
        Args:
            dataset : The input dataset to be processed. from phase_init
        '''
        semantic_dict = {
            'ss_sc' : [], # seen_skill, seen_context
            'ss_nc' : [], # seen_skill, novel_context
            'ns_sc' : [], # novel_skill, seen_context
            'ns_nc' : [], # novel_skill, novel_context
        }

        skill_group_context_probs ={}

        for idx in tqdm(range(len(dataset['skill_query']))) :
            skill_query = dataset['skill_query'][idx]
            context_query = dataset['context_query'][idx]

            skill_group_idx, skill_prob = self.retrieve_skill_group(skill_query)
            skill_group_idx = skill_group_idx.item()
            context_idx, context_prob = self.retrieve_context(context_query, np.array(skill_group_idx) ) 
            context_idx = context_idx.item()
            skill_group_prob = skill_prob[skill_group_idx] 
            context_match_prob = context_prob[context_idx]

            if skill_group_idx in skill_group_context_probs.keys() :
                skill_group_context_probs[skill_group_idx]['probs'].append(context_prob)
                skill_group_context_probs[skill_group_idx]['count'] += 1
            else :
                skill_group_context_probs[skill_group_idx] = {
                    'probs' : [context_prob],
                    'count' : 1,
                    'skill_query' : skill_query,
                }
            
            # seen detection strategy
            skill_seen = skill_group_prob > self.skill_threshold[skill_group_idx]
            context_seen = context_match_prob > self.context_threshold[context_idx]

            # dict_key = '{}s_{}c'.format('s' if skill_seen else 'n', 's' if context_seen else 'n')
            dict_key = '{}s_{}c'.format('s' if skill_seen else 'n', 's' if context_seen else 's')
            semantic_dict[dict_key].append((skill_group_idx, context_idx))

            if dict_key == 'ss_sc' :
                continue # for many of case
            elif dict_key == 'ss_nc' : 
                continue
                # for min thresholding 
                print( f"[{dict_key}] : {dataset['skills'][idx]}")
                self.expand_skill_group(
                    skill_group_idx=skill_group_idx,
                    context_idx=self.used_context_len,
                    skill_query=skill_query,
                    dataset=dataset,
                )
            elif dict_key == 'ns_sc' or dict_key == 'ns_nc' :
                if 'skills' in dataset.keys() :
                    print( f"[{dict_key}] : {dataset['skills'][idx]}, {self.skill_group_len}")
                else :
                    print( f"[{dict_key}] : {self.skill_group_len}")
                # if unused context and skill group is not full
                self.create_skill_group(
                    skill_group_idx=self.skill_group_len,
                    context_idx=self.used_context_len,
                    skill_query=skill_query,
                    dataset=dataset,
                )
                if 'skills' in dataset.keys() :
                    print( f"[{dict_key}] : {dataset['skills'][idx]},  {self.context_threshold[self.used_context_len-1]}")
                else :
                    print( f"[{dict_key}] : {self.context_threshold[self.used_context_len-1]}")

        for skill_group_idx in skill_group_context_probs.keys() :
            possible_contexts = self.skill_group[skill_group_idx].copy()
            # skill_group_context_probs[skill_group_idx]['sum'] /= skill_group_context_probs[skill_group_idx]['count']
            probs = np.vstack(skill_group_context_probs[skill_group_idx]['probs'])
            median_probs = np.median(probs, axis=0)
            skill_query = skill_group_context_probs[skill_group_idx]['skill_query']
            # compare each threshold. if novel context extend the group
            expand_flag = True
            for cid in possible_contexts :
                if median_probs[cid] > self.context_threshold[cid] :
                    expand_flag = False
            if expand_flag:
                print(f"skill group {skill_group_idx} is extended by context {self.used_context_len}")
                for cid in possible_contexts :
                    print(f"\t similarity of exist skill group {cid} vs threshold : {median_probs[cid]:.4f} vs {self.context_threshold[cid]:.4f}")
                self.expand_skill_group(
                    skill_group_idx=skill_group_idx,
                    context_idx=self.used_context_len,
                    skill_query=skill_query,
                    dataset=dataset,
                )
        unique_dict ={
            key : np.unique(np.array(semantic_dict[key]), axis=0) for key in semantic_dict.keys() 
        }
        print("unique dict : \n", unique_dict) # for logging debug
        return unique_dict

    def find_skill_group(self, dataset) :
        skill_group_indicies = np.zeros((len(dataset['skill_query'])), dtype=np.int32) # optimize
        jit_apply = jax.jit(self.skill_group_module.apply)
        for idx, skill_query in tqdm(enumerate(dataset['skill_query'])) : 
            prob = jit_apply(self.skill_group_module_train_state.params, skill_query)
            skill_group_idx = np.argmax(prob) # note!!
            skill_group_indicies[idx] = skill_group_idx
        return skill_group_indicies

    def find_context_matching(self, dataset) : 
        # context_indicies = None # get context indicies from dataset ( B, list of idxs)
        selected_indicies = [] # get selected indicies from context indicies ( B, 1)
        context_probs = [] # get context probs from context indicies ( B, P)
        for idx in tqdm(range(0, len(dataset['skill_group_idx']))) :
            skill_group_indicies = dataset['skill_group_idx'][idx]
            context_query = dataset['context_query'][idx]
            context_idx, context_prob = self.retrieve_context(context_query, np.array(skill_group_indicies) )
            selected_indicies.append(context_idx.copy()) 
            context_probs.append(context_prob.copy())
        selected_indicies = np.array(selected_indicies)
        context_probs = np.array(context_probs)
            
        reinit_dict = {} # (idx : params dictionary) used to update the module params
        
        dataset['context_idx'] = selected_indicies
        dataset['context_prob'] = context_probs
        return dataset, reinit_dict # data, skill_group_memory_idx , context_key_idx
  
    ## skill-hash key retrieval module ##
    @partial(jax.jit, static_argnums=(0,))
    def get_skill_prob_jit(self, params, query) :
        prob = self.skill_group_module.apply(params, query)
        skill_group_idx = jnp.argmax(prob, axis=-1)
        return skill_group_idx, prob

    def retrieve_skill_group(self, skill_embedding) :
        return self.get_skill_prob_jit(self.skill_group_module_train_state.params, skill_embedding)
        
    ## hash-context key retieval module ## 
    @partial(jax.jit, static_argnums=(0,))
    def get_context_prob_jit(self, params, query) :
        return self.context_module.apply(params, query)

    def retrieve_context(self, context_query, skill_group_idx=None) :
        prob = self.get_context_prob_jit(self.context_module_train_state.params, context_query)
        # balancing prob here
        if self.balance_context_prob :
            prob = prob * self.key_appear_prob_reverse
        
        if skill_group_idx is not None : # 0517 Edit here for unlearning!!!!
            if skill_group_idx.ndim == 0 :
                context_candidates = self.skill_group[skill_group_idx]
                mask = np.zeros(prob.shape[-1], dtype=bool)
                mask[context_candidates] = True
                prob = prob * mask
            elif skill_group_idx.ndim == 2 :
                mask = np.zeros(prob.shape, dtype=bool)
                for i, idx_set in enumerate(skill_group_idx):
                    context_candidates = self.skill_group[idx_set[0]]
                    mask[i,:,context_candidates] = True
                prob = prob * mask
        
        if self.retrieval_mode == 'sim' :
            context_idx = np.argmax(prob, axis=-1)
        elif self.retrieval_mode == 'nsim' :
            extended_treshold = self.context_threshold.copy()
            context_idx = np.argmax(prob-extended_treshold, axis=-1)
        else :
            raise NotImplementedError

        return context_idx, prob
    
    ## Util function ##
    @property   
    def key_appear_prob_reverse(self) : 
        # NOTE not used in dual hash model
        # conflict with thresholding method
        key_appear = self.key_appear_count_fixed 
        min_key_appear = np.min(key_appear)
        max_key_appear = np.max(key_appear)
        range_key_appear = max_key_appear - min_key_appear
        denominator = np.maximum(range_key_appear, 1)
        pre_softmax_prob = 1 - (key_appear - min_key_appear) / denominator
        return pre_softmax_prob 

## TODO Multikey operation integrated to DualHashModel
class DualHashMultiKeyModel(DualHashModel):
    def __init__(
            self,
            key_per_pool=20, # orig to 20
            **kwargs,
        ):
        self.key_per_pool = key_per_pool
        print(f"[DualHashMultiKeyModel] key_per_pool : {self.key_per_pool}")
        super().__init__(**kwargs)
    
    def init_context_match_module(self):
        self.context_module = MultiKeySimilarityModule(
            num_pool=self.num_keys, 
            key_num=self.key_per_pool,
            feature_dim=self.context_dim, 
        )
        context_module_params = self.context_module.init(
            jax.random.PRNGKey(0), 
            jnp.zeros((1, self.context_dim))
        )
        self.context_module_train_state = train_state.TrainState.create(
            apply_fn=self.context_module.apply, 
            params=context_module_params, 
            tx=self.cm_optimizer,
        )

    def expand_skill_group(
            self,
            skill_group_idx,
            context_idx,
            skill_query,
            dataset,
        ) : 
        if self.used_context_len >= self.num_keys :
            print(f"context is full! novel context is not added existing context will be selected.")
            return

        # get context indicies from dataset (maximum length k)
        skill_matched_idx = np.where(np.all(dataset['skill_query'] == skill_query, axis=1))[0]
        cq = dataset['context_query'][skill_matched_idx]

        # context_indicies = self.init_context_match(dataset, skill) 
        context_indicies = [context_idx] 
        params = self.context_module_train_state.params

        # normalized context anchor is set to (K, F) by k-means
        context_mean = KMeans(n_clusters=self.key_per_pool, random_state=0, n_init='auto').fit(cq).cluster_centers_

        normalized_context_anchor = context_mean / np.linalg.norm(context_mean, axis=-1, keepdims=True)    
        normalized_context = cq / np.linalg.norm(cq, axis=1, keepdims=True)

        # normalized_context (B, f) (K, f)
        sim_score = np.tensordot(normalized_context, normalized_context_anchor, axes=([-1], [-1]))
        sim_score = np.max(sim_score, axis=-1) # (B, 1)
        context_sim_min = np.median(sim_score) # (1)
        self.context_threshold[context_idx] = context_sim_min

        # keys (P, K, F)
        params['params']['keys'] = params['params']['keys'].at[context_idx].set(normalized_context_anchor)
        self.context_module_train_state = self.context_module_train_state.replace(params=params)
        self.skill_group[skill_group_idx].extend(context_indicies)

        self.used_context_len += 1

# lora book is implemented by dual_l2m
class ConsistencyHashModel(DualHashModel) :
    '''
    construct the dataset query cluster by output action dependency
    '''
    def __init__(
        self,
        **kwargs,
    ):
        self.key_per_pool = kwargs['memory_pool_config'].key_num
        self.action_dim = kwargs['memory_pool_config'].action_dim # TODO automate this by sample input of model mmworld 4, kitchen 9
        self.consistency_mode = kwargs['memory_pool_config'].consistency_mode # basic, noaction, onlyaction 
        self.tight_threshold = kwargs['memory_pool_config'].tight_threshold # default Fasle.
        print(f"[ConsistencyHashModel] key_per_pool : {self.key_per_pool}")
        super().__init__(**kwargs)
        self.init_action_consistency_module()
        self.first_phase = True
        self.learned_processing = kwargs['memory_pool_config'].learned_processing # 'delete' , 'reuse' # default 'basic'
        print(f"[ConsistencyHashModel] learned_processing : {self.learned_processing}")

    def init_context_match_module(self):
        self.context_module = MultiKeySimilarityModule(
            num_pool=self.num_keys, 
            key_num=self.key_per_pool,
            feature_dim=self.context_dim, 
        )
        context_module_params = self.context_module.init(
            jax.random.PRNGKey(0), 
            jnp.zeros((1, self.context_dim))
        )
        self.context_module_train_state = train_state.TrainState.create(
            apply_fn=self.context_module.apply, 
            params=context_module_params, 
            tx=self.cm_optimizer,
        )

    def init_action_consistency_module(self) :
        self.consistency_module = MultiKeySimilarityModule(
            num_pool=self.num_keys, 
            key_num=self.key_per_pool,
            feature_dim=self.action_dim, 
        )
        consistency_module = self.consistency_module.init(
            jax.random.PRNGKey(0), 
            jnp.zeros((1, self.action_dim))
        )
        self.consistency_module_train_state = train_state.TrainState.create(
            apply_fn=self.consistency_module.apply, 
            params=consistency_module, 
            tx=self.cm_optimizer,
        )
        self.consistency_threshold = np.ones((self.num_keys,), dtype=np.float32)
        
    ## core function ## 
    def process_dataset_key(self, dataset, model=None) :
        '''
        process dataset wit consistency state action matched key.
        '''
        if 'skill_query' not in dataset.keys() :
            raise ValueError(f'[skill_query] not in dataset keys {dataset.keys()}')
        if 'context_query' not in dataset.keys() :
            raise ValueError(f'[context_query] not in dataset keys {dataset.keys()}')
        self.used_context_len_prev = self.used_context_len

        ''' 
        # 1. split query by skill(shilluettte) and initialize the modulator
        # 1-2. match the query by existing model( like evaluation = argmax sim(q,k))
            # exception-first init (if first phase, hard initialization)
        # dataset : ['skill_group_idx', 'context_idx', 'key', 'probs'] added
        '''
        dataset, unique_dict = self.dataset_skill_groupping(dataset)
        dataset, module_reinit_dict = self.find_context_matching(dataset)
        
        '''
        # 2. evaluate the query by existing model( a ~ pi(a|s,d_id) )
        # dataset['pred_actions'] is created 
        '''
        pred_actions = np.zeros_like(dataset['actions'])
        batch = 1024
        for idx in tqdm(range(0, len(dataset['observations']), batch)) :
            obs = dataset['observations'][idx:idx+batch]
            if len(obs) == batch :
                actions = model.eval_model(obs[:,None,:])[:,0,:]
                pred_actions[idx:idx+batch] = actions
            else : # jax compile optimize
                ex_obs = np.concatenate([obs, np.zeros((batch-len(obs),obs.shape[-1]), dtype=np.float32)], axis=0)
                pred_actions[idx:idx+len(obs)] = model.eval_model(ex_obs[:,None,:])[:len(obs),0,:]
        dataset['pred_actions'] = pred_actions.copy()

        if self.first_phase : # first phase is not using consistency matching
            self.first_phase = False
            return dataset, module_reinit_dict
        '''
        # 3. dataset['pred_actions'] and action-prototype distribution matched by key
        # 3-1. [threshold calculation] if lower than threshold is modified to novel
            # dataset['key'] is modified if key == -1 then novel
        '''
        print("[consistency matching] start ")
        dataset = self.consistency_matching(dataset)
        
        '''
        # 4. if novel, then expand the skill group(new context) - novel key is fixed
            # if seen context, then do not modify the key
        '''
        dataset = self.create_novel_context(dataset)

        print("[consistency matching] total learnable key count")
        key_dict = {}
        for key in np.unique(dataset['key']) :
            if key == -1 :
                continue
            print(f"key {key} : {np.sum(dataset['key'] == key)}")
            key_dict[key] = np.sum(dataset['key'] == key)
        
        return dataset, module_reinit_dict

    def retrieve_key(self, query, **kwargs) :
        '''
        [Evaluation] only used in query evaluation 
            use super().retrieve_key
        '''
        return super().retrieve_key(query)

    def process_dataset_query(self, dataset) :
        '''
        [diff] process input query with output query
        seperate the action dataset query by.
        '''
        if type(dataset) == dict :
            dataset['skill_query'] = dataset['observations'][..., -512:]
            dataset['context_query'] = dataset['observations'][..., :self.context_dim]
            return dataset
        else : # for normal object
            raise NotImplementedError

    ## consistency mathching and thresholding ##
    def consistency_matching(self, dataset) :
        batch_size = 1024
        dataset_len = len(dataset['actions'])
        consistency_score = np.zeros((dataset_len,), dtype=np.float32)
        for idx in range(0, dataset_len, batch_size) :
            end_idx = min(idx + batch_size, dataset_len)
            pred_actions = dataset['pred_actions'][idx:end_idx]

            # Handling the last batch separately if it's not complete
            if len(pred_actions) < batch_size:
                pred_actions = np.concatenate([pred_actions, np.zeros((batch_size - len(pred_actions), pred_actions.shape[-1]))], axis=0)

            prob = self.get_consistency_prob(pred_actions)
            indices = dataset['key'][idx:end_idx]
            scores = prob[np.arange(len(indices)), indices]
            consistency_score[idx:end_idx] = scores[:len(indices)]

        dataset['consistency_score'] = consistency_score
        dataset['context_score'] = dataset['context_prob'][np.arange(len(dataset['context_idx'])), dataset['context_idx']]
        
        # novel calculation by consistency and state thresholding
        novel_consistency = np.where(dataset['consistency_score'] < self.consistency_threshold[dataset['key']])[0]
        novel_state = np.where(dataset['context_score'] < self.context_threshold[dataset['key']])[0]
        novel_index = np.unique(np.concatenate([novel_consistency, novel_state], axis=0))
        if self.consistency_mode == 'noaction' :
            print(f"[consistency matching] no consistency matching")
            novel_index = novel_state
        if self.consistency_mode == 'onlyaction' :
            print(f"[consistency matching] only consistency matching")
            novel_index = novel_consistency
        print(f"novel consistency : {len(novel_consistency)}")
        dataset['key'][novel_index] = -1
        return dataset 
    
    @partial(jax.jit, static_argnums=(0,))
    def get_consistency_prob_jit(self, params, query) :
        return self.consistency_module.apply(params, query)

    def get_consistency_prob(self, pred_actions) :
        prob = self.get_consistency_prob_jit(
            self.consistency_module_train_state.params,
            pred_actions,
        )
        return prob

    def create_novel_context(self, dataset) :
        print(f"[consistency matching] learned context with : {self.learned_processing}")

        unique_skill_group = np.unique(dataset['skill_group_idx'])
        use_indices = []
        if self.tight_threshold == True :
            dataset['key'] = np.ones((len(dataset['skill_group_idx']),), dtype=np.int32) * -1

        for skill_group_idx in unique_skill_group :
            skill_query = dataset['skill_query'][np.where(dataset['skill_group_idx'] == skill_group_idx)[0]][0]
            skill_indicies = np.where(dataset['skill_group_idx'] == skill_group_idx)[0]
            novel_context = np.where(dataset['key'] == -1)[0]
            novel_skill_context = np.intersect1d(skill_indicies, novel_context)



            if len(novel_skill_context) < self.key_per_pool and self.learned_processing != 'iscil' :
                dataset['key'][novel_skill_context] = -1 # dropped automatically in lorabook
                continue

            print(f"skill group {skill_group_idx} is extended by context {self.used_context_len}")
            
            # 0120 processing the key by learned processing
            if self.learned_processing == 'delete' :
                use_index = np.where(dataset['key'] == -1)[0]
                use_indices.append(use_index)
            elif self.learned_processing == 'reuse' or self.learned_processing == 'iscil' :
                dataset['key'][skill_indicies] = self.used_context_len
            elif self.learned_processing == 'basic' :
                dataset['key'][novel_skill_context] = self.used_context_len
            else :
                raise NotImplementedError

            self.expand_skill_group(
                skill_group_idx=skill_group_idx,
                context_idx=self.used_context_len,
                skill_query=skill_query,
                novel_data_idx=novel_skill_context,
                dataset=dataset,
            )

        if self.learned_processing == 'delete' :
            if len(use_indices) == 0 :
                for data_key in dataset.keys() :
                    use_indices = np.array([0,1], dtype=np.int32)
                    dataset[data_key] = dataset[data_key][use_indices]
            else :  
                use_indices = np.concatenate(use_indices, axis=0)
                if use_indices.shape[0] < self.num_keys :
                    print(f"novel context is not enough, {use_indices.shape[0]} < {self.num_keys}")
                    use_indices = np.array([0,1], dtype=np.int32)
                for data_key in dataset.keys() :
                    dataset[data_key] = dataset[data_key][use_indices] 
        return dataset 

    def expand_skill_group(
            self,
            skill_group_idx, # group index
            context_idx,  # novel context index
            # dataset
            skill_query=None,   
            novel_data_idx=None, 
            dataset=None,  
        ) :
        if self.used_context_len >= self.num_keys :
            print(f"context is full! novel context is not added existing context will be selected.")
            return
        
        context_indicies = [context_idx] 

        if novel_data_idx is None : # if novel data is not given, then use the skill query. (naive skill matching)
            novel_data_idx = np.where(np.all(dataset['skill_query'] == skill_query, axis=1))[0]
        cq = dataset['context_query'][novel_data_idx]
        if self.key_per_pool == 1 :
            context_mean = np.mean(cq, axis=0, keepdims=True)
        else : 
            context_mean = KMeans(n_clusters=self.key_per_pool, random_state=0, n_init='auto').fit(cq).cluster_centers_
        normalized_context_anchor = context_mean / np.linalg.norm(context_mean, axis=-1, keepdims=True) 
        normalized_context = cq / np.linalg.norm(cq, axis=1, keepdims=True)

        action = dataset['actions'][novel_data_idx]
        if self.key_per_pool == 1 :
            action_mean = np.mean(action, axis=0, keepdims=True)
        else :
            action_mean = KMeans(n_clusters=self.key_per_pool, random_state=0, n_init='auto').fit(action).cluster_centers_
        normalized_action_anchor = action_mean / np.linalg.norm(action_mean, axis=-1, keepdims=True) 
        normalized_action = action / np.linalg.norm(action, axis=1, keepdims=True)

        '''
        # 1. thresholding method part( min/median/mean ) TODO multi key
        '''
        sim_score = np.tensordot(normalized_context, normalized_context_anchor, axes=([-1], [-1]))
        sim_score = np.max(sim_score, axis=-1) # (B, 1)
        # context_sim_min = np.amin(sim_score)
        context_sim_min = np.mean(sim_score)
        self.context_threshold[context_idx] = context_sim_min

        sim_score = np.tensordot(normalized_action, normalized_action_anchor, axes=([-1], [-1]))
        sim_score = np.max(sim_score, axis=-1) # (B, 1)
        consistency_sim_min = np.mean(sim_score)
        self.consistency_threshold[context_idx] = consistency_sim_min

        '''
        # 2. context/consistency module update part
        '''
        params = self.context_module_train_state.params
        params['params']['keys'] = params['params']['keys'].at[context_idx].set(normalized_context_anchor)
        self.context_module_train_state = self.context_module_train_state.replace(params=params)
        
        params = self.consistency_module_train_state.params
        params['params']['keys'] = params['params']['keys'].at[context_idx].set(normalized_action_anchor)
        self.consistency_module_train_state = self.consistency_module_train_state.replace(params=params)

        self.skill_group[skill_group_idx].extend(context_indicies)
        self.used_context_len += 1

    ## phase processors ##   
    def update_skill_group(self, dataset) :
        '''
        Update the skill group by context matching(Novel semantic)
            seen_skill
                seen_context -> [diff] check the consistency
                novel_context -> [remove] skill expansion
            novel_skill
                seen_context -> existing skill 
                novel_context -> novel skill group generation
        Args:
            dataset : The input dataset to be processed. from phase_init
        '''
        semantic_dict = {
            'ss_sc' : [], # seen_skill, seen_context
            'ss_nc' : [], # seen_skill, novel_context
            'ns_sc' : [], # novel_skill, seen_context
            'ns_nc' : [], # novel_skill, novel_context
        }

        skill_group_context_probs ={}

        for idx in tqdm(range(len(dataset['skill_query']))) :
            skill_query = dataset['skill_query'][idx]
            context_query = dataset['context_query'][idx]

            skill_group_idx, skill_prob = self.retrieve_skill_group(skill_query)
            skill_group_idx = skill_group_idx.item()
            context_idx, context_prob = self.retrieve_context(context_query, np.array(skill_group_idx) ) 
            context_idx = context_idx.item()
            skill_group_prob = skill_prob[skill_group_idx] 
            context_match_prob = context_prob[context_idx]

            if skill_group_idx in skill_group_context_probs.keys() :
                skill_group_context_probs[skill_group_idx]['probs'].append(context_prob)
                skill_group_context_probs[skill_group_idx]['count'] += 1
            else :
                skill_group_context_probs[skill_group_idx] = {
                    'probs' : [context_prob],
                    'count' : 1,
                    'skill_query' : skill_query,
                }
            
            # seen detection strategy
            skill_seen = skill_group_prob > self.skill_threshold[skill_group_idx]
            context_seen = context_match_prob > self.context_threshold[context_idx]

            # dict_key = '{}s_{}c'.format('s' if skill_seen else 'n', 's' if context_seen else 'n')
            dict_key = '{}s_{}c'.format('s' if skill_seen else 'n', 's' if context_seen else 's')
            semantic_dict[dict_key].append((skill_group_idx, context_idx))

            if dict_key == 'ss_sc' :
                continue # for many of case
            elif dict_key == 'ss_nc' : 
                continue
            elif dict_key == 'ns_sc' or dict_key == 'ns_nc' :
                if 'skills' in dataset.keys() :
                    print( f"[{dict_key}] : {dataset['skills'][idx]}, {self.skill_group_len}")
                else :
                    print( f"[{dict_key}] : {self.skill_group_len}")
                # if unused context and skill group is not full
                self.create_skill_group(
                    skill_group_idx=self.skill_group_len,
                    context_idx=self.used_context_len,
                    skill_query=skill_query,
                    dataset=dataset,
                )
                if 'skills' in dataset.keys() :
                    print( f"[{dict_key}] : {dataset['skills'][idx]},  {self.context_threshold[self.used_context_len-1]}")
                else :
                    print( f"[{dict_key}] : {self.context_threshold[self.used_context_len-1]}")

        unique_dict ={
            key : np.unique(np.array(semantic_dict[key]), axis=0) for key in semantic_dict.keys() 
        }
        print("unique dict : \n", unique_dict) # for logging debug
        return unique_dict

    def find_context_matching(self, dataset) : 
        dataset , reinit_dict = super().find_context_matching(dataset)
        dataset['key'] = dataset['context_idx']
        dataset['probs'] = dataset['context_prob']
        return dataset, reinit_dict # data, skill_group_memory_idx , context_key_idx
    





if __name__ == '__main__' :
    from clus.env.continual_config import *
    from clus.env.cl_scenario import *
    from clus.env.metaworld_env import *
    # unit test for skill hash module

    dataloader_config = {
        'dataloader_cls' : MemoryPoolDataloader,
        'dataloader_kwargs' :{
            'skill_embedding_path' : 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl',
            'skill_exclude' : None,
            'semantic_flag' : True, 
        }
    }
    continual_scenario  = ContinualScenario(
        dataloader_config=dataloader_config,
        phase_configures=MM_EASY_TO_NORMAL_U72,
        # phase_configures=MM_EASY_TO_NORMAL_U24,
        evaluator=MMEvaluator(get_task_list_equal_normal(only_normal=True)[:1]),
    )
    skill_hash_module = DualHashModel()
    
    M = 20
    for i in range(24) : 
        skill_dataset = continual_scenario.get_phase_data(i).stacked_data
        skill_dataset['skill_query'] = skill_dataset['observations'][:, -512:]
        skill_dataset['context_query'] = skill_dataset['observations'][:, :140]
        for j in range(len(skill_dataset['context_query'])) :
            aggregate_start = max(j-M,skill_dataset['episode_boundary'][j][0])
            # print( skill_dataset['observations'][aggregate_start:j+1,:140].shape )
            aggregated_context = np.mean(skill_dataset['observations'][aggregate_start:j+1,:140], axis=0)
            # print( aggregated_context.shape)
            
            skill_dataset['context_query'][j] = aggregated_context
        skill_hash_module.process_dataset_key(skill_dataset)
    
    