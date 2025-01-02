
import cloudpickle
from tqdm import tqdm 
from clus.models.utils.train_state import *
from clus.models.model.basic import *
from clus.models.model.cdm import *
from clus.env.offline import *
from clus.env.metaworld_env import configs_task_list, get_task_list_equal_normal
# from clus.continual.utils import *
# from clus.continual.replay.generative_replay import *
from clus.utils.utils import create_directory_if_not_exists
from clus.env.cl_scenario import ContinualScenario,MultiTaskSceario
from clus.env.metaworld_env import MMEvaluator
from clus.env.kitchen import KitchenEvaluator
# from clus.trainer.base_trainer import BaseTrainer
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


import warnings
warnings.filterwarnings("ignore")
# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='L2M based continual learner trianing function.')
parser.add_argument('--save_id', type=str, default='2024-10-10-kitchen-comp-1')


default_path = './data/IsCiL_exp'
args = parser.parse_args()


def eval() : 
    args = parser.parse_args()
    test_id = args.save_id
    test_model_path = f'{default_path}/iscil/kitchen/{test_id}'

    dataloader_config = {
        'dataloader_cls' : MemoryPoolDataloader,
        'dataloader_kwargs' :{
            'skill_embedding_path' : 'data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',
            'skill_exclude' : None,
            'semantic_flag' : True, 
        }
    }   
    phase_configures = EK_COMPLETE
    continual_scenario = ContinualScenario(
        dataloader_config=dataloader_config,
        phase_configures=phase_configures,
        evaluator=KitchenEvaluator(
            phase_configures=phase_configures,
            eval_mode='obs',
            eval_episodes=3,
        ),
    )

    stage_eval_dict = {}
    for stage in range(len(EK_COMPLETE)) :
        print("\n\n================= Stage {} =================".format(stage))
        print(f'[Stage {stage} | Model loadeing] evaluating model_{stage}')
        with open(f'{test_model_path}/models/model_{stage}.pkl', 'rb') as f :
            model = cloudpickle.load(f)
            print(f'[Stage {stage} | Model loaded] evaluating model_{stage}')
            if stage == len(EK_COMPLETE)-1 :
                eval_dict = continual_scenario.evaluation_function(model, log_stage=stage, eval_episodes=5)
            else :
                eval_dict = continual_scenario.evaluation_function(model, log_stage=stage)
            stage_eval_dict[stage] = eval_dict
            print(stage_eval_dict)
            
    
    return stage_eval_dict

if __name__ == '__main__' :
    logging_path = f'/home/csitest/LifelongMultitask/logs'
    file_log = open(f'{logging_path}/{args.save_id}_eval.log', 'w')
    sys.stdout = DualStream(sys.stdout, file_log)
    
    print('Start evaluating model')
    stage_eval_dict = eval()
    with open(f'{logging_path}/{args.save_id}_metric.pkl', 'wb') as f :
        pickle.dump(stage_eval_dict, f)

    print('End evaluating model')
    file_log.close()
    sys.stdout = sys.__stdout__
    file_log.close()

    
                            