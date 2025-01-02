import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import jax 
import optax 
from dataclasses import dataclass, field
from flax import core
from flax import struct
from jax import tree_util
import numpy as np

# bidirectional adapter learning module 

def copy_lorabook_by_source_target(params, source, target):
    def copy_fn(path, p) :
        name = path[-1]
        if name == 'a' : # (book,N)
            return p.at[target,:].set(p[source,:].copy())
        elif name == 'b' :  # (M,book)
            return p.at[:,target].set(p[:,source].copy())
        else :
            return p
    return tree_util.tree_map_with_path(copy_fn, params)


def mask_lorabook_grads(grads, rank_mask):
    '''
    this function walks the leaf nodes of the grads and mask the gradients
    '''
    # rank_mask : (book,)
    def mask_fn (path, g) :
        name = path[-1]
        if name == 'a' : # (book,N)
            return g * rank_mask[:,None]
        elif name == 'b' : # (M,book)
            return g * rank_mask[None,:]
        else : 
            return g
    return tree_util.tree_map_with_path(mask_fn, grads)

class LoRABookTrainState(train_state.TrainState):
    '''
    TrainState for LoRABook Model
    '''
    train_mask_rank : struct.field(pytree_node=False) # index of masked rank (for gradient update)
    def apply_gradients(
        self,
        grads,
        **kwargs,
    ):  
        grads = mask_lorabook_grads(grads, self.train_mask_rank)
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
    
class EWCLoRABookTrainState(LoRABookTrainState):
    ewc_params : struct.field(pytree_node=True) # ewc params
    fisher_params : struct.field(pytree_node=True) # fisher params

    
### LoRABook Manager ###
class LoRABookManager():
    '''
    context, book link manager for LoRABook Model
    '''
    def __init__(
            self,
            book_size=64,
        ) -> None:
        self.book_size = book_size

        # book_used : (book_size,) 
        self.book_used = np.zeros(self.book_size) 

        # key : context, value : book_mask (book_size,)
        self.context_inference_book = {}
        self.context_learnable_book = {} 

    ### algorithm for book initialization ###
    def init_context(self, pool_idx, cprob=None):
        '''
        initialize the context inference book and context learnable bookinit
        consideration
        * learnable book selection 
        * inference book selection
        * modulator warm up mechanism ( knowledge initialization scheme )
        
        '''
        if pool_idx not in self.context_inference_book.keys() :
            # basic algorithm context and book is same
            first_unused_idx = int(np.sum(self.book_used))
            self.context_inference_book[pool_idx] = np.zeros(self.book_size)
            self.context_inference_book[pool_idx][first_unused_idx] = 1.0
            self.context_learnable_book[pool_idx] = np.zeros(self.book_size)
            self.context_learnable_book[pool_idx][first_unused_idx] = 1.0
            self.book_used[first_unused_idx] = 1.0
            print(f'init context {pool_idx} : {first_unused_idx},{self.context_inference_book[pool_idx]} , {self.context_learnable_book[pool_idx]}')
    
            if cprob is not None :
                pass
            # 
    def get_meta_init_list(self, dataset) :
        meta_init_list = []
        unique_keys = np.unique(dataset['key'])
        print(unique_keys)
        for key in unique_keys :
            pool_idx = int(key)
            if pool_idx in self.context_inference_book.keys() : 
                continue
            self.init_context(pool_idx)
            if 'probs' not in dataset.keys() :
                break
            non_zero_idx = np.nonzero(self.context_inference_book[pool_idx])[0]
            # get probs where 
            probs = dataset['probs'][dataset['key'] == key]
            mean_probs = np.mean(probs, axis=0)
            # get the second max probs of pool except for the key
            second_pool_idx = np.argsort(mean_probs)[-2]
            if second_pool_idx in self.context_inference_book.keys() :
                # get index of non-zero value
                second_nonzero_idx = np.nonzero(self.context_inference_book[second_pool_idx])[0]
                for j, idx in enumerate(second_nonzero_idx) :
                    if j > len(non_zero_idx) :
                        break
                    meta_init_list.append( {'source' : idx, 'target' : non_zero_idx[j]} )
    
        return meta_init_list
    
    ### get book mask ###
    def get_bookmask(self, pool_idx):
        '''
        (book_mask, rank_mask) pair is used for training
        book mask : (book_size,)
            used for inference module selection
        rank mask : (book_size,)
            used for learnable module selection on inference module
        '''
        if pool_idx not in self.context_inference_book.keys() :
            self.init_context(pool_idx, 0.0)

        book_mask = self.context_inference_book[pool_idx]
        grad_mask = self.context_learnable_book[pool_idx]
        
        return book_mask, grad_mask
    
class DyLoRABookManager(LoRABookManager) :
    def __init__(
            self,
            book_size=64, # pool_size * rank_size 
            rank_size=4,
            init_mode='reference',
            ref_dropout=0.0,
            train_nested_dropout=False,
            eval_process='basic',
        ) -> None:
        self.book_size = book_size
        self.rank_size = rank_size # default rank initialization size
        self.init_mode = init_mode # init mode for context inference book
        self.eval_process = eval_process # eval process for context inference book
        print(f"[DyLoRABookManager] eval_process : {self.eval_process}")

        self.ref_dropout = ref_dropout # dropout rate for reference dropout
        self.train_nested_dropout = train_nested_dropout # train nested dropout
        
        print('='*20)
        print(f'[DyLoRABookManager] book_size : {self.book_size}, rank_size : {self.rank_size}')
        print(f'[DyLoRABookManager] init_mode : {self.init_mode}')
        print('='*20)
        self.book_unused = [i for i in range(self.book_size)] # queue for book used

        '''
        alwayse important rank is located on front of the queue
        '''
        self.context_inference_book = {}  
        self.context_learnable_book = {}

    def init_context(self, pool_idx, cprob=None):
        '''
        initialize the context inference book and context learnable bookinit
        consideration
        * learnable book selection 
        * inference book selection
        * modulator warm up mechanism ( knowledge initialization scheme )
        
        '''
        if pool_idx not in self.context_inference_book.keys() :
            # select the idx for ranksize from unused queue
            book_pages = self.book_unused[:self.rank_size]
            self.book_unused = self.book_unused[self.rank_size:]
            
            book_pages = np.array(book_pages)
            self.context_inference_book[pool_idx] = book_pages.copy()
            self.context_learnable_book[pool_idx] = book_pages.copy()
            print(f'init context {pool_idx} : {book_pages}')

    def get_meta_init_list(self, dataset) :
        meta_init_list = []
        unique_keys = np.unique(dataset['key'])
        print("[lorabook meta_init]", unique_keys)
        for key in unique_keys :
            pool_idx = int(key)
            if pool_idx in self.context_inference_book.keys() : 
                continue
            self.init_context(pool_idx)
            if 'probs' not in dataset.keys() :
                break

            # get the second max probs of pool except for the key            
            init_pages = self.context_inference_book[pool_idx]
            probs = dataset['probs'][dataset['key'] == key]
            mean_probs = np.mean(probs, axis=0)
            second_pool_idx = np.argsort(mean_probs)[-2]

            '''
            There is 2 mechanism for meta initialization
            1. append context inference book with second_pool_idx
            2. initialize the learnable vector by second_pool_idx 
            '''
            if second_pool_idx in self.context_learnable_book.keys() : 
                reference_pages = self.context_learnable_book[second_pool_idx].copy()
                if self.init_mode == 'reference' : # # INIT MODE 1 full reference
                    self.context_inference_book[pool_idx] = np.concatenate([self.context_inference_book[pool_idx].copy(), reference_pages.copy()])

                elif self.init_mode == 'copy' :
                    for j, r_page in enumerate(reference_pages) : # # INIT MODE 2
                        if j > len(init_pages) :
                            break
                        meta_init_list.append( {'source' : r_page, 'target' : init_pages[j]} )

                elif self.init_mode == 'refcopy' : 
                    self.context_inference_book[pool_idx] = np.concatenate([self.context_inference_book[pool_idx].copy(), reference_pages.copy()])
                    for j, r_page in enumerate(reference_pages) :
                        if j > len(init_pages) :
                            break
                        meta_init_list.append( {'source' : r_page, 'target' : init_pages[j]} )
                else : # no meta init
                    pass
        return meta_init_list
    
    def get_bookmask(self, pool_idx, eval=False):
        '''
        (book_mask, rank_mask) pair is used for training
        book mask : (book_size,)
            used for inference module selection
        rank mask : (book_size,)
            used for learnable module selection on inference module
        '''
        if pool_idx not in self.context_inference_book.keys() :
            self.init_context(pool_idx, 0.0)

        # nested dropout for post-ranking dropout
        # TODO : deprecated on joint learning ###################
        # dropout for inference book
        dropout_var = np.random.rand()
        if dropout_var < self.ref_dropout and eval == False :
            book_mask = np.zeros(self.book_size)
            book_mask[self.context_learnable_book[pool_idx]] = 1.0
        elif self.eval_process == 'lonly' and eval == True :
            book_mask = np.zeros(self.book_size)
            book_mask[self.context_learnable_book[pool_idx]] = 1.0
        else :
            book_mask = np.zeros(self.book_size)
            book_mask[self.context_inference_book[pool_idx]] = 1.0
        grad_mask = np.zeros(self.book_size)
        grad_mask[self.context_learnable_book[pool_idx]] = 1.0
        # TODO : deprecated on joint learning ###################

        ## Dynamic learning for nested dropout
        if self.train_nested_dropout == True and eval == False :
            learnable_length = len(self.context_learnable_book[pool_idx])
            # nest range (0, learnable_length)
            nested_dropout_idx = np.random.randint(1, learnable_length)
            # print(self.context_inference_book[pool_idx][:nested_dropout_idx])
            grad_mask = np.zeros(self.book_size)
            grad_mask[self.context_learnable_book[pool_idx][:nested_dropout_idx]] = 1.0

            book_mask = np.zeros(self.book_size)
            book_mask[self.context_learnable_book[pool_idx][:nested_dropout_idx]] = 1.0

        return book_mask, grad_mask
    

    ### pruning based accumulation ###
    def prune_book_mask(self) :
        '''
        prune the book mask by the used book
        return : extend the unused bookmasks (re initialized by Training Module)
        '''

        ## algorithm for book mask

        # eval rank selection(all or partial) => plot the ranks and select the rank # 
        # 1. eval all ranks 
        # 2. find minimum value?(Not sure) affordable performance drop 

        # prune the book mask rank 
        pruned_bookmask = [1,2,3]

        self.book_unused.extend(pruned_bookmask)
        return pruned_bookmask 
        



