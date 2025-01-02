
from clus.env.cl_scenario import ContinualScenario 
from clus.env.metaworld_env import MMEvaluator

from clus.trainer.base_trainer import ContinualTrainer
from clus.env.continual_config import *
from clus.env.metaworld_env import get_task_list_equal_easy
import optax
from clus.models.utils.train_state import *
from clus.models.model.basic import *
from clus.models.model.cdm import *

# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='L2M based continual learner trianing function.')
parser.add_argument('-d', '--debug', type=bool, help='Set experiment to debug mode', default=False)
parser.add_argument('-e', '--env', type=str, help='mmworld, kitchen, libero', default='kitchen')
parser.add_argument('-ep', '--eval_episodes', type=int, help='default to 3', default=3)
parser.add_argument('-r', '--rank', type=int, help='Adapter basic rank 3', default=1)
args = parser.parse_args()

if __name__ == '__main__' :
    print('args' , args)
    debug_flag = args.debug
    env_type = args.env 
    eval_episodes = int(args.eval_episodes)
    rank = int(args.rank)

    optim_config = {
        'optimizer_cls' : optax.adam,
        'optimizer_kwargs' : {
            'learning_rate' : 1e-5 , 
            'b1' : 0.9,
        },
    } 

    diffusion_model_config = {
        'model_cls' : ConditionalDiffusion, 
        'model_kwargs' : {
            'input_config' : None,
            'optimizer_config' : optim_config,
            'model_config' : {
                'model_cls' :FlaxDenoisingBlockMLP,
                'model_kwargs' : {
                    'dim' : 512,
                    'n_blocks' : 4,
                    'context_emb_dim' : 512,
                    'dropout' : 0.0,
                }
            }, 
            'clip_denoised' : True,
            # 'diffusion_step' : 64,
            # 'diffusion_step' : 32,
        },
    }

    exp_config = {
        'phase_epoch' : 20000,
        'eval_epoch' : 5000,
        'batch_size' : 1024,
        'eval_env' : True,
        'base_path' : f'./data/pre/diffusion/{env_type}/deptest', # base path for saving items
        'phase_optim' : 're_initialize',
        'replay_method' : 'random',  
        'init_model_path' : None,
    }

    ## Continual Scenario
    dataloader_config = None
    from clus.env.offline import BaseDataloader
    if env_type == 'kitchen' :
        print("kitchen_evaluation")
        exp_config['phase_epoch'] = 50000
        exp_config['eval_epoch'] = 5000
        dataloader_config = {
                'dataloader_cls' : BaseDataloader,
                'dataloader_kwargs' :{
                    'skill_embedding_path' : 'data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',
                    'skill_exclude' : None,
                    'semantic_flag' : True, 
                }
            }   
        from clus.env.kitchen import KitchenEvaluator
        phase_configures = KITCHEN_MINIMAL
        continual_scenario = ContinualScenario(
            dataloader_config=dataloader_config,
            phase_configures=phase_configures,
            evaluator=KitchenEvaluator(
                phase_configures=phase_configures,
                eval_mode='obs',
                eval_episodes=10,
                semantic_flag=dataloader_config['dataloader_kwargs']['semantic_flag'],
            ),
        )
    elif env_type == 'mmworld' : 
        print("mmworld_evaluation")
        exp_config['phase_epoch'] = 10000
        exp_config['eval_epoch'] = 2000
        dataloader_config = {
            'dataloader_cls' : BaseDataloader,
            'dataloader_kwargs' :{
                'skill_embedding_path' : 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl',
                'skill_exclude' : None,
                'semantic_flag' : True, 
            }
        }   
        if debug_flag == False :
            continual_scenario = ContinualScenario(
                dataloader_config=dataloader_config,
                phase_configures=MM_EASY_0,
                evaluator=MMEvaluator(get_task_list_equal_easy(),
                    eval_mode='obs', 
                    eval_episodes=eval_episodes,
                ),
            )
    else :
        print( f"env_type : {env_type} is not supported")
        raise NotImplementedError
        

    trainer = ContinualTrainer(
        continual_scenario=continual_scenario,
        model_config=diffusion_model_config,
        exp_config=exp_config,
    )

    print(f'diffusion model config : {diffusion_model_config}')
    print(f'experiment config : {exp_config}')
    if dataloader_config is not None :
        print(f'dataloader config : {dataloader_config}')

    trainer.continual_train()