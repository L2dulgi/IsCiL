from clus.models.utils.train_state import *
from clus.models.model.basic import *
from clus.models.model.cdm import *
from clus.env.offline import *
from clus.env.metaworld_env import configs_task_list
from clus.utils.utils import create_directory_if_not_exists
from clus.env.cl_scenario import ContinualScenario
from clus.env.metaworld_env import MMEvaluator
from clus.env.kitchen import KitchenEvaluator
from clus.trainer.l2m_trainer import MemoryPoolCLTrainer
from clus.models.peftpool.dual_l2m import *
from clus.env.continual_config import *

import sys
class DualStream:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='L2M based continual learner trianing function.')
parser.add_argument('-d', '--debug', type=bool, help='Set experiment to debug mode', default=False)
parser.add_argument('-u', '--reinit', type=str, help='reinitialization? on every phase', default='re_initialize')

parser.add_argument('-e', '--env', type=str, help='mmworld, kitchen, libero', default='kitchen')
parser.add_argument('-ep', '--eval_episodes', type=int, help='default to 3', default=3)
parser.add_argument('-sp', '--spec', type=str, help='none 10 20', default=None)

parser.add_argument('-r', '--rank', type=int, help='Adapter basic rank 1', default=1)
parser.add_argument('-s', '--shot', type=int, help='shot count per each dataset', default=None)
parser.add_argument('-al', '--algo', type=str, help='pet_pool algorithm', default='l2m')
parser.add_argument('-id', '--save_id', type=str, help='save path under default path', default='test')
parser.add_argument('-seed', '--seed', type=int, help='make seed for all experiments', default=None)
parser.add_argument('-epoch', '--epoch', type=int, help='full_epoch per dataset(for cw10)', default=3000)
parser.add_argument('-lr', '--lr', type=float, help='learning rate', default=5e-4)
parser.add_argument('-k', '--key_num', type=int, help='key_num', default=1)


default_path = './data/IsCiL_exp'
args = parser.parse_args()

if __name__ == '__main__' :

    if args.seed is not None :  
        np.random.seed(args.seed)
        random.seed(args.seed)

    # args processing
    debug_flag = args.debug
    env_type = args.env 
    eval_episodes = int(args.eval_episodes)
    rank = int(args.rank)
    algo = args.algo
    key_num = args.key_num
    lr = args.lr
    reinit = args.reinit

    few_shot_len = args.shot

    full_path = f"{default_path}/{args.algo}/{args.env}/{args.save_id}"
    logging_path_base = f"{full_path}/training_log.log"

    if debug_flag == True :
        full_path = f"{default_path}/debug"
        logging_path_base = f"{default_path}/debug/training_log.log"
    else :
        create_directory_if_not_exists(full_path)
        # create log_file

    ### model and optimzer configures ### 
    optim_config = {
        'optimizer_cls' : optax.adam,
        'optimizer_kwargs' : {
            'learning_rate' : lr, 
        },
    } 
    diffusion_model_config = None

    ### experiment configures ### 
    peft_pool_model_cls=LoRAPoolModel
    exp_config = {
        'phase_epoch' : 2000,
        'eval_epoch' : 2000,
        'batch_size' : 1024,
        'eval_env' : True if eval_episodes > 0 else False,
        'base_path' : full_path, # base path for saving items
        'phase_optim' : reinit,
        'replay_method' : 'random',  
        'phase_batch_sz' : None, # for ER mmworld e2m
        'init_model_path' : 'data/pre_trained_models/evolving_world/diffusion/model_0.pkl',
    }

    ### Continual Scenario ### 
    dataloader_config = {
        'dataloader_cls' : MemoryPoolDataloader,
        'dataloader_kwargs' :{
            'skill_embedding_path' : 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl',
            'skill_exclude' : None,
            'semantic_flag' : True, 
        }
    }
    state_dim = 0
    action_dim = 4
    continual_scenario = None
    if env_type == 'kitchen' :
        print("kitchen_evaluation")
        exp_config['phase_epoch'] = args.epoch
        exp_config['eval_epoch'] = args.epoch 
        exp_config['init_model_path'] = 'data/pre_trained_models/evolving_kitchen/diffusion/model_0.pkl'
        state_dim=572
        action_dim=9

        scenario_cls = ContinualScenario
        phase_configures = EK_COMPLETE
        if args.spec == 'complete' :
            phase_configures = EK_COMPLETE
        elif args.spec == 'semi' :
            phase_configures = EK_SEMI
        elif args.spec == 'incomplete' :
            phase_configures = EK_INCOMPLETE
        else :
            print("not supported spec")
            raise NotImplementedError

        dataloader_config = {
                'dataloader_cls' : MemoryPoolDataloader,
                'dataloader_kwargs' :{
                    'skill_embedding_path' : 'data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',
                    'skill_exclude' : None,
                    'semantic_flag' : True, 
                    ## for small data per dataset ##
                    'few_shot_len' : few_shot_len,
                }
            }   
        if args.spec != 'semi' :
            continual_scenario = ContinualScenario(
                dataloader_config=dataloader_config,
                phase_configures=phase_configures,
                evaluator=KitchenEvaluator(
                    phase_configures=phase_configures,
                    eval_mode='obs',
                    eval_episodes=3,
                ),
            )
        else : # this is semi
            continual_scenario = ContinualScenario(
                dataloader_config=dataloader_config,
                phase_configures=phase_configures,
                evaluator=KitchenEvaluator(
                    phase_configures=phase_configures[:10],
                    eval_mode='obs',
                    eval_episodes=3,
                ),
            )
    elif env_type == 'mmworld' : 
        print("mmworld_evaluation")
        exp_config['phase_epoch'] = args.epoch
        exp_config['eval_epoch'] = args.epoch
        state_dim = 652
        action_dim=4
        phase_configures=MM_EASY_TO_HARD_M

        if args.spec == 'complete' :
            phase_configures = MW_COMPLETE
        elif args.spec == 'semi' :
            phase_configures = MW_SEMI_COMPLETE
        elif args.spec == 'incomplete' :
            phase_configures = MW_INCOMPLETE
        else :
            print("not supported spec")
            raise NotImplementedError
        
        if args.spec == 'semi' or args.spec == '20hs2' :
            continual_scenario = ContinualScenario(
                dataloader_config=dataloader_config,
                phase_configures=phase_configures,
                evaluator=MMEvaluator(configs_task_list(phase_configures[:10]),
                    eval_mode='obs', 
                    eval_episodes=eval_episodes,
                    phase_configures=phase_configures[:10],
                ),
            )
        else :
            continual_scenario = ContinualScenario(
                dataloader_config=dataloader_config,
                phase_configures=phase_configures,
                evaluator=MMEvaluator(configs_task_list(phase_configures),
                    eval_mode='obs', 
                    eval_episodes=eval_episodes,
                    phase_configures=phase_configures,
                ),
            )
    else :
        print( f"env_type : {env_type} is not supported")
        raise NotImplementedError

    

    ## Modulation methods ##
    from clus.config.algo_config import get_algorithm_configs
    algorithm = get_algorithm_configs(algo=algo)
    algorithm.peft_pool_model_kwargs['memory_pool_config'].feature_dim=state_dim
    algorithm.peft_pool_model_kwargs['memory_pool_config'].lora_dim=rank
    algorithm.peft_pool_model_kwargs['memory_pool_config'].key_num=key_num
    algorithm.peft_pool_model_kwargs['memory_pool_config'].action_dim=action_dim

    if env_type == 'cw10' : 
        algorithm.peft_pool_model_kwargs['context_mode'] = 'cw10'
    
    algorithm.peft_pool_model_kwargs['lora_optimizer_config'] = optim_config

    if args.seed is not None :  
        np.random.seed(args.seed)
        random.seed(args.seed)

    ### start the experiment ### 
    if debug_flag == True :
        exp_config['phase_epoch'] = 2
        exp_config['eval_epoch'] = 2
        # exp_config['eval_env'] = False
        exp_config['eval_env'] = True

    trainer = MemoryPoolCLTrainer(
        algorithm=algorithm,
        continual_scenario=continual_scenario,
        model_config=diffusion_model_config,
        exp_config=exp_config,
    )

    logging_path = logging_path_base
    file_log = open( logging_path, "w")
    dual_stream = DualStream(sys.stdout, file_log)
    sys.stdout = dual_stream

    print('args' , args)
    print('[dataset] few_shot_len per dataset', few_shot_len)
    print(f'experiment config : {exp_config}')
    if dataloader_config is not None :
        print(f'dataloader config : {dataloader_config}')
    trainer.continual_train()

    sys.stdout = sys.__stdout__
    file_log.close()