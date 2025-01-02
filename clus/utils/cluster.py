'''
this file is used to validate the cluster data
'''

from clus.env.kitchen import KitchenEvaluator
from clus.env.cl_scenario import ContinualScenario 
from clus.env.continual_config import *
from clus.env.offline import *
from clus.models.peftpool.skill_hash import ContextHashModel, DualHashModel, DualHashMultiKeyModel
from clus.models.peftpool.pool_basis import LoRAMemoryPoolConfig



if __name__ == "__main__" :
    # load Continual Scenario
    dataloader_config = {
        'dataloader_cls' : BaseDataloader,
        'dataloader_kwargs' :{
            'skill_embedding_path' : 'data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',
            'skill_exclude' : None,
            'semantic_flag' : True, 
        }
    }

    phase_configures = KITCHEN_MINIMAL_TO_FULL_24
    continual_scenario = ContinualScenario(
        dataloader_config=dataloader_config,
        phase_configures=phase_configures,
        evaluator=KitchenEvaluator(
            phase_configures=phase_configures,
            eval_mode='obs',
            eval_episodes=3,
        ),
    )
    memory_pool_config=LoRAMemoryPoolConfig(
        pool_length=100,
        # feature_dim=652, # mmworld
        feature_dim=572, # kitchen
        lora_dim=1,
        embedding_key='split'
    )
    # set the HashKey module 
    hash_key_module = DualHashMultiKeyModel(
        memory_pool_config=memory_pool_config,
        context_dim=memory_pool_config.feature_dim-512,
        context_mode='obs',
        retrieval_mode='sim',
    )
    ## for loop for continual leanring module ## 
    # check each data is clustered well by accuracy and text logging

    
    datasets = [] 

    for phase in range(continual_scenario.phase_num) :
        print(f"\n\n")
        print(f"phase {phase} : {continual_scenario.phase_configures[phase]['data_name']}")
        dataloader = continual_scenario.get_phase_data(phase)
        stacked_data = hash_key_module.process_dataset_query(dataloader.stacked_data)
        processed_data, _ = hash_key_module.process_dataset_key(stacked_data)
        datasets.append(processed_data.copy())

        # count 
        for j in range(phase) : 
            s_d, _ = hash_key_module.process_dataset_key(datasets[j])
            key_unique = np.unique(s_d['key'])
            print(f"[phase{phase}] prev phase {j} : {key_unique}")
        
    # [] visualize the cluster accuracy 
    # [] visualize the training selection 

    # visualize the evaluation selection (Use the ready made kithen data)
    # actually save the data by (task - skill)

    # analyze the performance forgetting of model

    # comparison normal, l2m, and ours
    pass