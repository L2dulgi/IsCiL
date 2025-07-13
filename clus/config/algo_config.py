# mlp diffusion 
from clus.models.model.basic import *
from clus.models.model.cdm import *
from clus.models.peftpool.dual_l2m import *

from clus.models.peftpool.pool_basis import *
from clus.models.peftpool.skill_hash import ContextHashModel, ConsistencyHashModel, DualHashMultiKeyModel

class ExpAlgorithmConfig() :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls =  LoRAPoolModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=100,
                feature_dim=512, # TODO updated by outer space 
                lora_dim=1,
                embedding_key='split'
            ),
            'key_module_cls' : ContextHashModel,
            # 'key_module_cls' : DualHashMultiKeyModel,
            'context_mode':'obs', 
            'retrieval_mode':'sim',
            'meta_init' : False,
        }

class ExpSeqLoRA(ExpAlgorithmConfig) :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls =  LoRAPoolModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=1,
                feature_dim=512, 
                lora_dim=64, # 128 updated in 0114
                embedding_key='split'
            ),
            'key_module_cls' : ContextHashModel,
            'context_mode':'obs', 
            'retrieval_mode':'sim',
        }

class ExpTAIL(ExpAlgorithmConfig) :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls =  LoRAPoolModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=1,
                feature_dim=512, 
                lora_dim=16, # 128 updated in 0114
                embedding_key='split'
            ),
            'key_module_cls' : ContextHashModel,
            'context_mode':'obs', 
            'retrieval_mode':'sim',
            'tail_flag' : True,
        }

class ExpL2M(ExpAlgorithmConfig) :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls =  LoRAPoolModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=100,
                feature_dim=512, 
                lora_dim=4,
                embedding_key='split'
            ),
            'key_module_cls' : ContextHashModel,
            'context_mode':'obs', 
            'retrieval_mode':'sim',
            'key_mode' : 'single',
            'learnable_key' : True,
        }

class ExpL2MBASE(ExpAlgorithmConfig) :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls =  LoRAPoolModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=100,
                feature_dim=512, 
                lora_dim=4,
                embedding_key='base'
            ),
            'key_module_cls' : ContextHashModel,
            'context_mode':'obs', 
            'retrieval_mode':'sim',
            'key_mode' : 'single',
            'learnable_key' : True,
        }

class ExpTAILG(ExpAlgorithmConfig) :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls =  LoRAPoolModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=8,
                feature_dim=512, 
                lora_dim=16,
                embedding_key='tailg'
            ),
            'key_module_cls' : ContextHashModel,
            'context_mode':'obs', 
            'retrieval_mode':'sim',
            'key_mode' : 'tailg',
            'learnable_key' : False,
        }

class EepIsCIL(ExpAlgorithmConfig) :
    def __init__(
        self,
    ): 
        self.peft_pool_model_cls = DyLoRABookModel
        self.peft_pool_model_kwargs = {
            'model' : None, 
            'memory_pool_config': LoRAMemoryPoolConfig(
                pool_length=100,
                feature_dim=512, 
                lora_dim=4,
                embedding_key='split',
                meta_init_mode='copy',
                learned_processing='iscil', 
            ),
            'key_module_cls' : ConsistencyHashModel, 
            'context_mode':'obs', 
            'retrieval_mode':'sim',
            'meta_init' : True,
        }



        
def get_algorithm_configs(algo='l2m') :
    if algo == 'l2m' :
        return ExpL2MBASE()
    if algo == 'l2mg' :
        return ExpL2M()
    if algo == 'tailg' :
        return ExpTAILG()
    elif algo == 'tail' :
        return ExpTAIL()
    elif algo == 'seqlora' :
        return ExpSeqLoRA()
    elif algo == 'iscil' :
        return EepIsCIL()
    else :
        print(f'algo {algo} is not supported')
        raise NotImplementedError
    
dif_transformer = {
    'model_cls' : ConditionalDiffusion,
    'model_kwargs' : {
        'input_config' : None,
        'optimizer_config' : None,
        'model_config' : {
            'model_cls' :FlaxDenoisingBlock,
            'model_kwargs' : {
                'dim' : 512,
                'n_blocks' : 4,
                'n_heads' : 2,
                'd_head' : 128,
                'context_emb_dim' : 512,
                'dropout' : 0.1,
                'only_self_attention' : False,
            }
        }, 
        'clip_denoised' : False,
        'diffusion_step' : 64,
    },
}
