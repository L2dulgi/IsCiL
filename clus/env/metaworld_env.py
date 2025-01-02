
import pickle
import numpy as np
from contextlib import contextmanager
from itertools import permutations, product
from clus.models.peftpool.dual_l2m import DyLoRABookModelOracle



# <<<<<<<<<< multi-stage metaworld >>>>>>>>>> #
def get_task_list_equal_easy(all_task_flag='full'):
    task_sets = [['puck', 'drawer', 'button', 'door']] # easy_door 
    ss =[0, 3, 4, 7, 8, 11]  # all task cover | half
    bound = 12 # hyper parameter for task boundary
    if all_task_flag == 'full' : 
        ss = [i for i in range(12)]
    if all_task_flag == 'half' : 
        ss = [0, 3, 4, 7, 8, 11] 
    if all_task_flag == 'third' : 
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == "sixth":
        ss = [0, 10, 13, 23]
        bound = 24
    task_shuffled = []
    count = 0
    for task_set in task_sets :
        for i, d in enumerate(permutations(task_set)):
            d = list(d)
            if i%bound  in ss :
                task_dict = {
                    'skill_list' : list(task_set),
                    'skill_seq' : list(d) 
                }
                task_shuffled.append( task_dict )
            count += 1
        # input()
    return task_shuffled # list of ( skill_list , skill_seq ) dictionary

def get_task_list_equal_normal(all_task_flag='full', only_normal=False):
    task_sets=[]

    print( f"tasks normal\n" * 1 )
    for i in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        if 'box' in i or 'stick' in i :
            continue
        if only_normal :
            if 'handle' not in i and 'lever' not in i :
                continue
        task_sets.append(list(i))

    task_shuffled = []
    count = 0
    ss =[0, 3, 4, 7, 8, 11]  # all task cover | half
    bound = 12 # hyper parameter for task boundary
    if all_task_flag == 'full' : 
        ss = [i for i in range(12)]
    if all_task_flag == 'half' : 
        ss = [0, 3, 4, 7, 8, 11] 
    if all_task_flag == 'third' : 
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == "sixth":
        ss = [0, 10, 13, 23]
        bound = 24

    for task_set in task_sets :
        for i, d in enumerate(permutations(task_set)):
            d = list(d)
            if i%bound in ss : #NOTE
                task_dict = {
                    'skill_list' : list(task_set),
                    'skill_seq' : list(d) 
                }
                task_shuffled.append( task_dict )
            count += 1
        # input()
    return task_shuffled # list of ( skill_list , skill_seq ) dictionary

def get_task_list_equal_hard(all_task_flag='full'):
    task_sets=[]

    print( f"tasks hard\n" * 1 )
    for i in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        task_sets.append(list(i))

    task_shuffled = []
    count = 0
    ss =[0, 3, 4, 7, 8, 11]  # all task cover | half
    bound = 12 # hyper parameter for task boundary
    if all_task_flag == 'full' : 
        ss = [i for i in range(12)]
    if all_task_flag == 'half' : 
        ss = [0, 3, 4, 7, 8, 11] 
    if all_task_flag == 'third' : 
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == "sixth":
        ss = [0, 10, 13, 23]
        bound = 24

    for task_set in task_sets :
        for i, d in enumerate(permutations(task_set)):
            d = list(d)
            if i%bound in ss : #NOTE
                task_dict = {
                    'skill_list' : list(task_set),
                    'skill_seq' : list(d) 
                }
                task_shuffled.append( task_dict )
            count += 1
    return task_shuffled # list of ( skill_list , skill_seq ) dictionary


def configs_task_list(configs) :
    task_shuffled = get_task_list_equal_hard()
    for task in task_shuffled :
        task['data_name'] = "-".join(task['skill_seq'])

    task_refined = []
    for phase in configs :
        for tasks in phase['data_name'].split(',') :
            if type(tasks) == str :
                tasks = [tasks]

            for task in tasks :
                # find task in task_shuffle
                for task_dict in task_shuffled :
                    if task == task_dict['data_name'] :
                        task_refined.append(task_dict)
                        break
                
    return task_refined

from tqdm import tqdm

try: 
    from mmworld.envs.mujoco.sawyer_xyz.v2.sawyer_non_stationary_v2 import SawyerNonStationaryEnvV2
except :
    print("mmworld not installed")
import random as py_rand
import gym
from clus.env.base_evaluator  import BaseEvaluator
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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

from clus.env.continual_config import *

class MMEvaluator(BaseEvaluator) :
    def __init__(
            self,
            base_evaluation_sequences,
            eval_mode='obs', # obs or traj
            traj_length=10, # used for traj eval mode
            eval_episodes=3,
            phase_configures=None,
        ) -> None:

        print("[MMevaluator]")
        skill_embedding_path='data/continual_dataset/evolving_world/mm_lang_embedding.pkl'
        with open( skill_embedding_path , 'rb' ) as f :
            self.skill_semantics = pickle.load(f)
        
        self.eval_horizons = 600

        self.base_evaluation_sequences = base_evaluation_sequences
        self.threshold = len(self.base_evaluation_sequences)
        self.eval_mode = eval_mode
        self.traj_length = traj_length
        self.eval_episodes = eval_episodes

        self.env_list = []
        for idx , task in enumerate(tqdm(self.base_evaluation_sequences)) :
            # env initialize
            skill_list = task['skill_list']
            env = SingleTask(seed=777, skill_list=skill_list) # From m-metaworld
            env.env.skill_list = task['skill_seq']
            skill_seq = task['skill_seq'] 
            self.env_list.append(env)
            if len(self.env_list) < self.threshold : 
                continue
        
        if phase_configures is not None :
            self.phase_configures = phase_configures
            for config in tqdm(self.phase_configures) :
                for path in config['data_paths'] :
                    with open(path, 'rb') as f :
                        data = pickle.load(f)
                    ep_done = np.where(np.array(data['terminals']) == 1)[0][0]
                    first_traj = np.array(data['observations'])[:ep_done]

                    skills = np.array(data['skills'])[:ep_done]
                    skill_embs = np.array([self.skill_semantics[i] for i in skills])
                    first_traj = np.concatenate([first_traj, skill_embs], axis=-1)
 
    def evaluate_base(
        self,
        model,
        eval_fn=None,
        task_model=None,
        log_stage=None,
        eval_episodes=None,
    ):
        """
        Parameters
        ----------
        model : object
            The model (or wrapper) to be evaluated.
        eval_fn : callable, optional
            If specified, use this function for evaluation. If not specified,
            use model.eval_model by default.
        task_model : object, optional
            If there is a separate 'task model' (e.g., for multi-task training),
            specify it here. If provided, this function will forward the evaluation
            to self.task_eval(...) and return that result.
        log_stage : int, optional
            If you want to log only up to a certain stage index, specify it here.
        eval_episodes : int, optional
            Number of episodes to evaluate. If not specified, self.eval_episodes is used.
        """

        # 1) If a separate task_model is provided, forward to self.task_eval
        if task_model is not None:
            return self.task_eval(model, task_model, eval_fn)

        # Determine the number of evaluation episodes
        if eval_episodes is None:
            eval_episodes = self.eval_episodes

        # Dictionary to store the results
        rew_info = {
            'skill_seq': [],  # Stores the skill sequence of each environment
            'skill_rew': [],  # Stores the aggregated reward (or success metric) for each environment
        }

        # Set eval_fn to model.eval_model if not explicitly provided
        eval_fn = model.eval_model if eval_fn is None else eval_fn

        used_unique = []

        # Loop over multiple evaluation episodes
        for eval_seed in range(eval_episodes):
            # Initialization at the start of each episode
            history_obs_list = []
            obs_list = []
            done_list = []
            skill_idx_list = []

            # Reset each environment
            for e_idx, env in enumerate(self.env_list):
                obs = env.reset()
                obs_list.append(obs)
                done_list.append(False)
                skill_idx_list.append(0)

            # A dummy observation for environments that have ended
            dummy_obs = np.zeros_like(obs_list[0])

            # Iterate over the time horizon
            for _ in tqdm(range(self.eval_horizons), desc=f"[Eval Seed: {eval_seed}]"):
                skill_semantics_list = []
                # Construct skill semantics input if needed
                for e_idx, env in enumerate(self.env_list):
                    sidx = min(skill_idx_list[e_idx], 3)

                    # If the environment has env.env.skill_list
                    if hasattr(env, "env") and hasattr(env.env, "skill_list"):
                        current_skill = env.env.skill_list[sidx]
                        if hasattr(self, "skill_semantics"):
                            skill_semantics = self.skill_semantics[current_skill]
                        else:
                            # If self.skill_semantics is not available, just use a placeholder
                            skill_semantics = np.array([sidx], dtype=np.float32)
                    else:
                        # If the environment does not have skill_list, use a fallback
                        current_skill = sidx
                        skill_semantics = np.array([sidx], dtype=np.float32)

                    skill_semantics_list.append(skill_semantics)

                # Construct the observation for the model (with or without semantics)
                if getattr(self, "semantic_flag", True):
                    obs_input = np.concatenate([obs_list, skill_semantics_list], axis=-1)
                else:
                    obs_input = np.array(obs_list)

                # Call the model to get actions
                unique = None
                if self.eval_mode == 'obs':
                    # Simply call eval_fn on obs_input
                    eval_res = eval_fn(obs_input[:, None, :])

                    # Post-processing of the returned result
                    if isinstance(eval_res, tuple):
                        actions, unique = eval_res
                    else:
                        actions = eval_res

                    actions = np.array(actions)  # shape: (N, action_dim)

                elif self.eval_mode == 'traj':
                    # Keep a trajectory of observations
                    if len(history_obs_list) == 0:
                        history_obs_list = np.tile(obs_input[:, None, :], (1, self.traj_length, 1))
                    else:
                        history_obs_list = np.concatenate([history_obs_list, obs_input[:, None, :]], axis=1)

                    # If the trajectory is too long, use only the latest part
                    if history_obs_list.shape[1] > self.traj_length:
                        history_obs_list = history_obs_list[:, -self.traj_length:, :]

                    actions, unique = eval_fn(history_obs_list)
                    actions = np.array(actions)
                else:
                    raise ValueError(f"eval_mode {self.eval_mode} is not defined")

                # Record unique values if any
                if unique is not None:
                    used_unique.append(unique)

                # Step through each environment with the chosen actions
                obs_list = []
                # If needed, slice actions down to a certain dimension
                if actions.shape[-1] > 4:
                    actions = actions[..., :4]

                for e_idx, env in enumerate(self.env_list):
                    # If the environment is already done, feed dummy_obs
                    if done_list[e_idx]:
                        obs_list.append(dummy_obs)
                        continue

                    next_obs, rew, done, env_info = env.step(actions[e_idx].squeeze())

                    obs_list.append(next_obs)

                    # If the environment indicates success, increment skill index
                    if 'success' in env_info and env_info['success'] == 1:
                        skill_idx_list[e_idx] += 1

                    # Check done status
                    if done:
                        done_list[e_idx] = True

                # Break if all environments are done
                if all(done_list):
                    break

            # Episode is finished; record skill sequence and rewards
            for env in self.env_list:
                # Example: gather skill_list and mode from env.env
                if hasattr(env, "env"):
                    skill_seq = getattr(env.env, "skill_list", [])
                    reward_sum = int(getattr(env.env, "mode", 0))  # Or any other measure if needed
                else:
                    skill_seq = []
                    reward_sum = 0

                if eval_seed == 0:
                    rew_info['skill_seq'].append(skill_seq)
                    rew_info['skill_rew'].append(reward_sum)
                else:
                    idx = rew_info['skill_seq'].index(skill_seq)
                    rew_info['skill_rew'][idx] += reward_sum

            # Clear lists at the end of an episode
            obs_list = []
            done_list = []
            skill_idx_list = []

        # After all episodes, calculate and print the results
        total_reward = 0.0
        for i, data in enumerate(rew_info['skill_seq']):
            rew_info['skill_rew'][i] /= eval_episodes
            print(
                f"[{i}] skill_seq: {rew_info['skill_seq'][i]}, "
                f"avg rew: {rew_info['skill_rew'][i]:.2f}"
            )
            print("[task {}] sub_goal sequence is {} task GC : {:.2f}% ({:.2f} / 4.00)".format(
                i,rew_info['skill_seq'][i], rew_info['skill_rew'][i]/4*100, rew_info['skill_rew'][i]))
            total_reward += rew_info['skill_rew'][i]

            # If log_stage is specified and matches i, stop logging here
            if log_stage is not None and log_stage == i:
                break

        # If log_stage is not None, we print the average up to that stage
        if log_stage is not None:
            avg_stage_reward = total_reward / (log_stage + 1)
            print("\n[Stage Multi-task(Total) GC] : {:.2f}% ({:.2f} / 4.00)".format(
                avg_stage_reward/4*100, avg_stage_reward))
        # If we collected any unique values, print them
        if len(used_unique) > 0:
            print("unique:", np.unique(np.concatenate(used_unique)))

        return rew_info
                    
        # def evaluate_base(
        #         self,
        #         model,
        #         eval_fn = None,
        #         log_stage=None,
        #     ) :
        #     eval_episodes = self.eval_episodes
        #     rew_info = {'skill_seq':[], 'skill_rew' : []}
        #     daco_flag = True if type(model) == DyLoRABookModelOracle else False
        #     eval_fn = model.eval_model if eval_fn is None else eval_fn
        #     used_unique = []
        #     unique = None
        #     # obs processing 
        #     for eval_seed in range(eval_episodes) :
        #         # reset the environment
        #         history_obs_list = []
        #         obs_list = []
        #         done_list= []
        #         skill_idx_list=[]
        #         for e_idx, env in enumerate(self.env_list) :
        #             obs = env.reset()
        #             obs_list.append(obs)
        #             done_list.append(False)
        #             skill_idx_list.append(0)

        #         dummy_obs = np.zeros_like(obs_list[0])
                
        #         for _ in tqdm(range(self.eval_horizons)) :
        #             skill_semantics_list = []
        #             for e_idx, env in enumerate(self.env_list) :
        #                 sidx = min(skill_idx_list[e_idx],3)
        #                 skill_semantics_list.append(self.skill_semantics[env.env.skill_list[sidx]])

        #             obs = np.concatenate([obs_list, skill_semantics_list], axis=-1)

        #             if self.eval_mode == 'obs' :
        #                 if daco_flag :
        #                     eval_res = eval_fn(obs[:,None,:], daco_query=self.daco_query)
        #                 else :
        #                     eval_res = eval_fn(obs[:,None,:])
        #                 # post processing
        #                 if type(eval_res) == tuple :
        #                     actions , unique = eval_res
        #                 else :
        #                     actions = eval_res
        #                 actions = np.array(actions) # mmworld action space
        #             elif self.eval_mode == 'traj' :
        #                 history_obs_list = np.concatenate([history_obs_list, obs[:,None,:]],axis=1) \
        #                     if len(history_obs_list) > 0 else np.tile(obs[:,None,:], (1,self.traj_length,1))
        #                 if len(history_obs_list) > self.traj_length :
        #                     history_obs_list = history_obs_list[:, -self.traj_length:, :]
        #                 actions, unique = eval_fn(history_obs_list)
        #                 actions = np.array(actions)
        #             else :
        #                 raise ValueError(f"eval_mode {self.eval_mode} is not defined")

        #             if unique is not None :
        #                 used_unique.append(unique)

        #             obs_list = []
        #             actions = actions[...,:4]
        #             for e_idx, env in enumerate(self.env_list) :
        #                 # pass if done
        #                 if done_list[e_idx] is True:
        #                     obs_list.append(dummy_obs) # dummy
        #                     continue
        #                 obs, rew, done, env_info = env.step(actions[e_idx].squeeze())
                        
        #                 obs_list.append(obs)
        #                 if env_info['success'] == 1 :
        #                     skill_idx_list[e_idx] += 1
        #                     if done :
        #                         done_list[e_idx] = True
        #             if done_list.count(True) == len(self.env_list) :
        #                 break

        #         for env in self.env_list :
        #             skill_seq = env.env.skill_list
        #             reward_sum = int(env.env.mode)
        #             if eval_seed == 0 :
        #                 rew_info['skill_seq'].append(skill_seq)
        #                 rew_info['skill_rew'].append(reward_sum)
        #             else : 
        #                 rew_info['skill_rew'][rew_info['skill_seq'].index(skill_seq)] += reward_sum
                
        #         obs_list = []
        #         done_list= []
        #         skill_idx_list=[]
        #     # eval episodes for loop end
        #     reward_sum = 0
        #     for i , data in enumerate(rew_info['skill_seq']) :
        #         rew_info['skill_rew'][i] /= eval_episodes
        #         print("[{}]skill is  {} rew : {:.2f}".format(i,rew_info['skill_seq'][i], rew_info['skill_rew'][i]))
        #         reward_sum += rew_info['skill_rew'][i]
            
        #     print("total reward : ", reward_sum/len(rew_info['skill_seq']))
        #     if len(used_unique) > 0 :
        #         print("unique : ", np.unique(np.concatenate(used_unique)))

        #     eval_reward = reward_sum/len(rew_info['skill_seq'])
        #     return rew_info
