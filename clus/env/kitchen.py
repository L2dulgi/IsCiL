import mujoco_py
import numpy as np
from d4rl.kitchen.kitchen_envs import KitchenBase, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from clus.env.continual_config import *
from clus.models.peftpool.dual_l2m import DyLoRABookModelOracle
from contextlib import contextmanager
import gym
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle

class KitchenTask:
    def __init__(self, subtasks):
        for subtask in subtasks:
            if subtask not in all_tasks:
                raise ValueError(f'{subtask} is not valid subtask')
        self.subtasks = subtasks

    def __repr__(self):
        return f"MTKitchenTask({' -> '.join(self.subtasks)})"
    
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

class KitchenEnv(KitchenBase):
    render_width = 1024
    render_height = 1024
    render_device = 0

    def __init__(self, *args, **kwargs):
        self.TASK_ELEMENTS = all_tasks  # for initialization
        self.TASK_ELEMENTS_TODO = all_tasks  # for initialization
        super().__init__(*args, **kwargs)
        self.task = None
        # self.TASK_ELEMENTS = all_tasks #  04
    
    def set_task_default(self,task) :
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        # default goal task infomation of kitchen-mixed-v0
        subtasks = [ 'microwave', 'kettle', 'bottom burner', 'light switch']
        trained_task = KitchenTask(
            subtasks=subtasks,
        )
        print("Semantic Skill Seq : " , task)
        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        
        self.task = trained_task
        self.TASK_ELEMENTS = trained_task.subtasks
        self.TASK_ELEMENTS_TODO = task.subtasks
        self.tasks_to_complete = task.subtasks

    @contextmanager
    def set_task(self, task):
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        # default goal task infomation of kitchen-mixed-v0
        subtasks = [ 'microwave', 'kettle', 'bottom burner', 'light switch']
        trained_task = KitchenTask(
            subtasks=subtasks,
        )
        print("Semantic Skill Seq : " , task)
        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        
        self.task = trained_task
        self.TASK_ELEMENTS = trained_task.subtasks
        self.TASK_ELEMENTS_TODO = task.subtasks
        self.tasks_to_complete = task.subtasks
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements
        self.tasks_to_complete = prev_task_elements
        self.TASK_ELEMENTS_TODO = prev_task_elements
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        self.render_width = width
        self.render_height = height
        self.render_device = device
        self.metadata['video.frames_per_second'] = fps
        self.metadata['video.frame_drop'] = frame_drop

    def _get_task_goal_todo(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS_TODO:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def compute_reward(self, obs_dict):
        reward_dict = {}
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']

        next_goal = self._get_task_goal_todo() 
        
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete and all_completed_so_far:
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        for completion in completions:
            self.tasks_to_complete.remove(completion)
        
        reward = float(len(completions))
        return reward
    
    def reset_model(self):
        ret = super().reset_model()
        self.tasks_to_complete = list(self.TASK_ELEMENTS_TODO)
        return ret # ret

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        if not self.initializing:
            a = self.act_mid + a * self.act_amp

        self.robot.step(self, a, step_duration=self.skip * self.model.opt.timestep)

        obs = self._get_obs()
        reward = self.compute_reward(self.obs_dict)
        done = not self.tasks_to_complete
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
        }
        return obs, reward, done, env_info

    def render(self, mode='rgb_array'):
        return super().render(mode='rgb_array')
        
from clus.env.base_evaluator  import BaseEvaluator

class KitchenEvaluator(BaseEvaluator):
    def __init__(
            self,
            phase_configures=KITCHEN_MINIMAL_TO_FULL_24,
            eval_mode='obs',
            traj_length=10,
            eval_episodes=1,
            semantic_flag=True,
        ) -> None:
        print("[kitchen evaluator init]")

        self.eval_mode=eval_mode
        self.traj_length=traj_length
        self.eval_episodes=eval_episodes
        self.phase_configures = phase_configures
        self.semantic_flag = semantic_flag


        self.skill_evaulation_path ='data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl'
        
        self.initial2task_dict = {
            'm' : 'microwave',
            'k' : 'kettle',
            'b' : 'bottom burner',
            't' : 'top burner',
            'l' : 'light switch',
            's' : 'slide cabinet',
            'h' : 'hinge cabinet',  
        }

        self.evaluation_sequences = []
        for idx , phase_configure in enumerate(self.phase_configures) :
            tasks = phase_configure['data_name'].split(',')
            for tid , task in enumerate(tasks) :
                tid = ''.join([t[0] for t in task.split('-')])               
                
                if 'domain' in phase_configure.keys() :
                    self.evaluation_sequences.append({
                        'task' : tid,
                        'domain' : phase_configure['domain'] # TODO consider domain mixed data
                    }) 
                else :        
                    self.evaluation_sequences.append({
                        'task' : tid
                    })

        self.possible_evaluation = [
            'mtlh','mlsh','mktl','mkth','mksh',
            'mkls','mklh','mkbs','mkbh','mbts',
            'mbtl','mbth','mbsh','mbls','ktls',
            'klsh','kbts','kbtl','kbth','kbsh',
            'kbls','kblh','btsh','btls',
        ]
        with open( self.skill_evaulation_path , 'rb' ) as f :
            self.skill_embedding = pickle.load(f)
        self.eval_horizons = 350

        self.env_list = []
        self.task_list = []
        self.domain_configs = []
        for idx , taks_configs in enumerate(self.evaluation_sequences) :
            env = KitchenEnv()
            test_task = KitchenTask(
                subtasks=self.initial2task(taks_configs['task']),
            )
            env.set_task_default(test_task)
            self.env_list.append(env)
            self.task_list.append(test_task)
            self.domain_configs.append(taks_configs['domain'] if 'domain' in taks_configs.keys() else None) 
            print("task : ", test_task.subtasks, " domain : ", taks_configs['domain'] if 'domain' in taks_configs.keys() else None)
        self.domain_setting()


    def domain_setting(self) :
        self.domain_scale = []
        self.domain_shift = []

        for idx , domain_config in enumerate(self.domain_configs) :
            if domain_config is None :
                self.domain_scale.append(np.ones((30,)))
                self.domain_shift.append(np.zeros((30,)))
            else :
                self.domain_scale.append(domain_config['scale'].copy())
                self.domain_shift.append(domain_config['shift'].copy())

        self.domain_scale = np.array(self.domain_scale)
        self.domain_shift = np.array(self.domain_shift)
        print("domain scale : ", self.domain_scale.shape)
        print("domain shift : ", self.domain_shift.shape)

    def domain_processing(self, states, eid=None):
        # action dataset is scaled by  (scale*a + shift)
        # evlauation is done by reverse process (a - shift) / scale
        original_states = states.copy()
        mode_len = len(self.domain_scale[0])
        ret_states = states
        if states.ndim == 1 :
            ret_states[:mode_len] = states[:mode_len].copy()*self.domain_scale[eid] + self.domain_shift[eid]
        else :
            ret_states[...,:mode_len] = states[...,:mode_len].copy()*self.domain_scale[eid] + self.domain_shift[eid]
        
        return ret_states

    def initial2task(
            self,
            initial=None
        ) : # initial : str ex) 'mkbh'
        if initial is None :
            return None
        return [ self.initial2task_dict[i] for i in initial ]   

    def task_eval(
        self,
        model,
        task_model=None,
        eval_fn=None,
    ) :
        eval_episodes = self.eval_episodes
        rew_info = {'skill_seq': [], 'skill_rew' : []}

        eval_fn = model.eval_model if eval_fn is None else eval_fn
        used_unique = []
        for eval_seed in range(eval_episodes) :
            obs_list = []
            done_list= []
            skill_idx_list=[]
            episode_reward = np.zeros((len(self.env_list),))

            for e_idx , env in enumerate(self.env_list) :
                obs = env.reset()
                mod_obs = self.domain_processing(obs,eid=e_idx)
                obs_list.append(mod_obs)
                done_list.append(False)
                skill_idx_list.append(0)

            dummy_obs = np.zeros_like(obs_list[0])

            for _ in tqdm(range(self.eval_horizons)) :    
                skill_semantics_list = []
                for e_idx, env in enumerate(self.env_list) :
                    task_obs = min(int(episode_reward[e_idx]),3)
                    skill_semantic = task_model(self.task_list[e_idx], task_obs)
                    skill_semantics_list.append(skill_semantic)
                task_obs = min(int(episode_reward[e_idx]),3)

                obs = np.concatenate([obs_list, skill_semantics_list], axis=-1)

                unique = None
                eval_res = eval_fn(obs[:,None,:], task_obs)
                if type(eval_res) == tuple :
                    actions , unique = eval_res
                else :
                    actions = eval_res
                actions = np.array(actions)
                
                if unique is not None :
                    used_unique.append(unique)

                obs_list = []
                for e_idx, env in enumerate(self.env_list) :
                    # pass if done
                    if done_list[e_idx] is True:
                        obs_list.append(dummy_obs)
                        continue

                    obs, rew, done, env_info = env.step(actions[e_idx].squeeze())
                    obs = self.domain_processing(obs,eid=e_idx).copy()
                    obs_list.append(obs)
                
                    episode_reward[e_idx] += rew
                    if done :
                        done_list[e_idx] = True
                if done_list.count(True) == len(self.env_list) :
                    break
            
            for eid , env in enumerate(self.env_list) :
                skill_seq = self.task_list[eid].subtasks
                reward_sum = episode_reward[eid]
                if eval_seed == 0 :
                    rew_info['skill_seq'].append(skill_seq)
                    rew_info['skill_rew'].append(reward_sum)
                else : 
                    rew_info['skill_rew'][rew_info['skill_seq'].index(skill_seq)] += reward_sum
            obs_list = []
            done_list= []
            skill_idx_list=[]

    ## evaluation 
    def evaluate_base(
            self,
            model,
            eval_fn=None,
            task_model=None,
            log_stage=None,
            eval_episodes=None,
        ) :
        # forwarding task evaluation for other function
        if task_model is not None :
            return self.task_eval(model, task_model, eval_fn)
        
        if eval_episodes is None :
            eval_episodes = self.eval_episodes
        rew_info = {'skill_seq': [], 'skill_rew' : []}
        daco_flag = True if type(model) == DyLoRABookModelOracle else False
        eval_fn = model.eval_model if eval_fn is None else eval_fn
        used_unique = []
        for eval_seed in range(eval_episodes) :
            history_obs_list = []
            obs_list = []
            done_list= []
            skill_idx_list=[]
            episode_reward = np.zeros((len(self.env_list),))

            for e_idx , env in enumerate(self.env_list) :
                obs = env.reset()
                mod_obs = self.domain_processing(obs,eid=e_idx)
                obs_list.append(mod_obs)
                done_list.append(False)
                skill_idx_list.append(0)

            dummy_obs = np.zeros_like(obs_list[0])

            for _ in tqdm(range(self.eval_horizons)) :
                skill_semantics_list = []
                for e_idx, env in enumerate(self.env_list) :
                    sidx = min(int(episode_reward[e_idx]),3)
                    skill_semantics_list.append(self.skill_embedding[self.task_list[e_idx].subtasks[sidx]])
                
                if self.semantic_flag == True :
                    obs = np.concatenate([obs_list, skill_semantics_list], axis=-1)
                else :
                    obs = np.array(obs_list)

                ## action prediction 
                unique = None
                if self.eval_mode == 'obs' :
                    if daco_flag :
                        eval_res = eval_fn(obs[:,None,:], daco_query=self.daco_query)
                    else :
                        eval_res = eval_fn(obs[:,None,:])
                    # post processing
                    if type(eval_res) == tuple :
                        actions , unique = eval_res
                    else :
                        actions = eval_res
                    actions = np.array(actions)
                elif self.eval_mode == 'traj' :
                    history_obs_list = np.concatenate([history_obs_list, obs[:,None,:]],axis=1) \
                        if len(history_obs_list) > 0 else np.tile(obs[:,None,:], (1,self.traj_length,1))
                    if len(history_obs_list) > self.traj_length :
                        history_obs_list = history_obs_list[:, -self.traj_length:, :]
                    actions, unique = eval_fn(history_obs_list)
                    actions = np.array(actions)
                else :
                    raise ValueError(f"eval_mode {self.eval_mode} is not defined")
            
                if unique is not None :
                    used_unique.append(unique)

                obs_list = []
                for e_idx, env in enumerate(self.env_list) :
                    # pass if done
                    if done_list[e_idx] is True:
                        obs_list.append(dummy_obs)
                        continue

                    obs, rew, done, env_info = env.step(actions[e_idx].squeeze())
                    obs = self.domain_processing(obs,eid=e_idx).copy()
                    obs_list.append(obs)
                
                    episode_reward[e_idx] += rew
                    if done :
                        done_list[e_idx] = True
                if done_list.count(True) == len(self.env_list) :
                    break
            
            for eid , env in enumerate(self.env_list) :
                skill_seq = self.task_list[eid].subtasks
                reward_sum = episode_reward[eid]
                if eval_seed == 0 :
                    rew_info['skill_seq'].append(skill_seq)
                    rew_info['skill_rew'].append(reward_sum)
                else : 
                    rew_info['skill_rew'][rew_info['skill_seq'].index(skill_seq)] += reward_sum
            obs_list = []
            done_list= []
            skill_idx_list=[]
                        
        reward_sum = 0
        # print("========= Result of Evaluation =========")
        for i , data in enumerate(rew_info['skill_seq']) :
            rew_info['skill_rew'][i] /= eval_episodes
            print("[task {}] sub_goal sequence is {} task GC : {:.2f}% ({:.2f} / 4.00)".format(
                i,rew_info['skill_seq'][i], rew_info['skill_rew'][i]/4*100, rew_info['skill_rew'][i]))
            reward_sum += rew_info['skill_rew'][i]
            if log_stage == i :
                break

        if log_stage is None :
            pass
        else : 
            print("\n[Stage Multi-task(Total) GC] : {:.2f}% ({:.2f} / 4.00)".format(reward_sum/(log_stage+1)/4*100, reward_sum/(log_stage+1)))
        if len(used_unique) > 0 :
            print("unique : ", np.unique(np.concatenate(used_unique)))

        if log_stage is None :
            return rew_info
        return rew_info['skill_rew'][:log_stage+1]
    