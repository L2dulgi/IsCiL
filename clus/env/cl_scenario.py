import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import cloudpickle
from tqdm import tqdm
try: 
    from mmworld.envs.mujoco.sawyer_xyz.v2.sawyer_non_stationary_v2 import SawyerNonStationaryEnvV2
except : 
    pass
import random as py_rand
class SingleTask(gym.Env):
    def __init__(self, seed: int, skill_list, obs_type='sensor', max_episode_length=1000, partially_observable=False):
        py_rand.seed(seed)
        self.env = SawyerNonStationaryEnvV2(skill_list)

        self.max_episode_length = max_episode_length
        self.time_steps = 0

        self.obs_type = obs_type
        self.partially_observable = partially_observable

        if self.obs_type == 'vision':
            self.observation_space = gym.spaces.Box(low=np.zeros((80, 80, 3)), high=np.ones((80, 80, 3)), dtype=np.uint8)
            self.env._partially_observable = self.partially_observable
        if self.obs_type == 'mixed':
            self.env._partially_observable = True
            self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(low=np.zeros((80, 80, 3)), high=np.ones((80, 80, 3)), dtype=np.uint8),
                                      'sensor': self.env.observation_space})
        elif self.obs_type == 'sensor':
            self.env._partially_observable = self.partially_observable
            self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space

    def step(self, action, action_noise=None):
        '''
        action : normalized action (-1, 1)
        action_noise : action noise
        '''
        sensor_obs, reward, done, info = None, None, None, None
        if action_noise is not None :
            sensor_obs, reward, done, info = self.env.step(action, action_noise)
        else :
            sensor_obs, reward, done, info = self.env.step(action)
        self.time_steps += 1

        if self.time_steps == self.max_episode_length:
            done = True

        if self.obs_type == 'vision':
            obs = self.render()
        elif self.obs_type == 'mixed':
            image_obs = self.render()
            obs = {'image': image_obs, 'sensor': sensor_obs}
        else:
            obs = sensor_obs

        info['action_noise'] = action_noise
        return obs, reward, done, info

    def reset(self):

        sensor_obs = self.env.reset()
        self.time_steps = 0
        if self.obs_type == 'vision':
            obs = self.render()
        elif self.obs_type == 'mixed':
            image_obs = self.render()
            obs = {'image': image_obs, 'sensor': sensor_obs}
        else:
            obs = sensor_obs
        return obs

    def render(self, mode='corner3', resolution=(224,224)):
        return self.env.render(offscreen=True, resolution=resolution, camera_name=mode)

DEFAULT_CS = [
    {
        'data_name' : 'puck',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/easy/puck.pkl',
        ],
    },
    {
        'data_name' : 'button',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/easy/button.pkl',
        ],
    },
    {
        'data_name' : 'door',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/easy/door.pkl',
        ],
    },
    {
        'data_name' : 'drawer',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/easy/drawer.pkl',
        ],
    },
]  

import numpy as np
from clus.env.offline import BaseDataloader,TemporalDataloader
from clus.env.metaworld_env import get_task_list_equal_easy, get_task_list_equal_hard, get_task_list_equal_normal
from clus.env.metaworld_env import MMEvaluator

class ContinualScenario() :
    '''
    class for continual learning scenario 
    contains eeach phase data information
    '''
    def __init__(
            self,
            phase_configures:list=None,
            dataloader_config=None,
            evaluator=None,

        ) -> None:
        
        self.phase_configures = phase_configures
        if self.phase_configures is None :
            self.phase_configures = DEFAULT_CS

        self.phase_num = len(self.phase_configures)

        self.evaluator = evaluator
        if self.evaluator is None :
            self.evaluator = MMEvaluator(get_task_list_equal_easy()[:1])
        self.evaluation_function = self.evaluator.evaluate_base

        
        self.dataloader_config = dataloader_config
        if self.dataloader_config is None :
            self.dataloader_config = {
                'dataloader_cls' : BaseDataloader,
                'dataloader_kwargs' :{
                    'skill_embedding_path' : 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl',
                    'skill_exclude' : None,
                    'semantic_flag' : True, 
                }
            }   
        ## params of each phase
        self.phase_idx = 0 

        self.init_input_config()

    def init_input_config(self) :
        self.dataloader_config['dataloader_kwargs']['data_paths'] = self.phase_configures[0]['data_paths']
        dummy_dataloader = self.dataloader_config['dataloader_cls'](
            **self.dataloader_config['dataloader_kwargs'],
        )
        demo_dataset = dummy_dataloader.get_rand_batch(4)
        self.data_feature_configs ={}
        for k in demo_dataset.keys() :
            if type(demo_dataset[k][0]) is np.ndarray :
                self.data_feature_configs[k] = demo_dataset[k][0].shape[-1]
        print(f'[data feature config] {self.data_feature_configs}')

    def get_phase_data(self, phase_idx) :
        self.phase_idx = phase_idx
        # NOTE 
        self.dataloader_config['dataloader_kwargs']['data_paths'] = self.phase_configures[self.phase_idx]['data_paths']
        if 'domain' in self.phase_configures[phase_idx].keys() :
            self.dataloader_config['dataloader_kwargs']['domain_info'] = self.phase_configures[phase_idx]['domain']
        if 'skill_exclude' in self.phase_configures[phase_idx].keys() :
            self.dataloader_config['dataloader_kwargs']['skill_exclude'] = self.phase_configures[phase_idx]['skill_exclude']

        dataloader = self.dataloader_config['dataloader_cls'](
            **self.dataloader_config['dataloader_kwargs'],
        )
        self.current_phase_dataloader = dataloader
        return dataloader     
    
class MultiTaskSceario(ContinualScenario) :
    def __init__(
            self,
            **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.origin_phase_num = len(self.phase_configures)
        self.phase_num = 1

    def get_phase_data(self, phase_idx) :
        self.phase_idx = phase_idx
        
        multi_task_dataloader = None
        for i in range(self.origin_phase_num) :
            print(f"[MultiTaskSceario] get_phase_data {i}")
            self.dataloader_config['dataloader_kwargs']['data_paths'] = self.phase_configures[i]['data_paths']
            if 'domain' in self.phase_configures[i].keys() :
                self.dataloader_config['dataloader_kwargs']['domain_info'] = self.phase_configures[i]['domain']
            
            dataloader = self.dataloader_config['dataloader_cls'](
                **self.dataloader_config['dataloader_kwargs'],
            )
            if 'domain' in self.phase_configures[i].keys() :
                del self.dataloader_config['dataloader_kwargs']['domain_info']
            if i == 0 :
                multi_task_dataloader = dataloader
            else :
                multi_task_dataloader.merge(dataloader)
            print(f"[MultiTaskSceario] get_phase_data {i} done {multi_task_dataloader.stacked_data['observations'].shape}")

        return multi_task_dataloader