
import cloudpickle


from clus.env.cl_scenario import ContinualScenario,MultiTaskSceario
from clus.env.continual_config import *
from clus.env.kitchen import KitchenEvaluator
from clus.env.metaworld_env import configs_task_list, get_task_list_equal_normal
from clus.env.offline   import MemoryPoolDataloader
from clus.utils.utils import create_directory_if_not_exists


try:
    from clus.env.metaworld_env import MMEvaluator
except :
    pass

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

class UnseenEvaluator() :
    def __init__(
        self,
        continual_scenario=None,
        
    ) :
        self.continual_scenario  = continual_scenario      
        self.eval_function = self.continual_scenario.evaluation_function

    def eval_algo(self, base_path) :
        # optimize for model load and key loading 
        for phase in range(20) :
            model_name = f'/models/model_{phase}.pkl'
            # model_name = f'/models/model_19.pkl'
            ### important for phase split
            print(f'phase {phase}')
            print(continual_scenario.phase_configures[phase])
            eval_dict = self.evaluate(phase, base_path, model_name)

    def evaluate(self, phase, model_path, model_name) :
        with open(model_path + model_name, 'rb') as file:
            self.model = cloudpickle.load(file)


        eval_dict = self.eval_function(self.model)
        return eval_dict

import argparse
parser = argparse.ArgumentParser(description='L2M based continual learner trianing function.')
parser.add_argument('-e', '--env', type=str, help='mmworld, kitchen, libero', default='kitchen')
parser.add_argument('-al', '--algo', type=str, help='algorithm', default='iscil')
parser.add_argument('-u', '--unseen', type=str, help='unseen type', default='ius')
parser.add_argument('-id', '--id', type=str, help='evaluation id', default='HelloIsCiL_complete_0')
args = parser.parse_args()



evaluation_id_list =[

]

if __name__ == "__main__" : 
    # read the file from target path
    model_path = 'data/IsCiL_exp/iscil/kitchen/HelloIsCiL_complete_0/models/'
    dataloader_config = {
        'dataloader_cls' : MemoryPoolDataloader,
        'dataloader_kwargs' :{
            'skill_embedding_path' : 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl',
            'skill_exclude' : None,
            'semantic_flag' : True, 
        }
    }

    env_type = args.env
    unseen = args.unseen
    if env_type == 'mmworld' :
        phase_configures = MM_COMPLETE_IND  
        if unseen == 'ius' :
            phase_configures = MM_COMPLETE_IUS
        else: 
            raise NotImplementedError
        continual_scenario = ContinualScenario(
            dataloader_config=dataloader_config,
            phase_configures=phase_configures,
            evaluator=MMEvaluator(configs_task_list(phase_configures),
                eval_mode='obs', 
                eval_episodes=3,
                phase_configures=phase_configures,
            ),
        )
    elif env_type == 'kitchen' :
        dataloader_config['dataloader_kwargs']['skill_embedding_path'] = 'data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl'
        phase_configures = EK_COMPLETE_IND
        continual_scenario = ContinualScenario(
            dataloader_config=dataloader_config,
            phase_configures=phase_configures,
            evaluator=KitchenEvaluator(
                eval_mode='obs', 
                eval_episodes=3,
                phase_configures=phase_configures,
            ),
        )
    else :
        raise NotImplementedError
    
    log_base_path = f'data/Unseen_experiments/{args.algo}/{env_type}/{unseen}'

    # open the model_file by model index.
    eval_path_base = 'data/IsCiL_exp' if args.algo not in ['seq', 'ewc', 'mtseq'] else 'data/seq_expriments'
    eval_path_base = f'{eval_path_base}/{args.algo}/{env_type}/{args.id}'
    
    logging_path_base = f'{log_base_path}/{args.id}' 
    create_directory_if_not_exists(logging_path_base)
    logging_path = f'{logging_path_base}/training_log.log'
    file_log = open( logging_path, "w")
    dual_stream = DualStream(sys.stdout, file_log)
    sys.stdout = dual_stream

    evaluator = UnseenEvaluator(
        continual_scenario=continual_scenario
    )

    evaluator.eval_algo(eval_path_base)
    
    sys.stdout = sys.__stdout__
    file_log.close()




