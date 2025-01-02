from clus.env.continual_config import *
from clus.env.offline import MemoryPoolDataloader
from clus.env.cl_scenario import ContinualScenario
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from clus.env.metaworld_env import get_task_list_equal_easy, get_task_list_equal_hard, get_task_list_equal_normal
from clus.env.metaworld_env import MMEvaluator

if __name__ == "__main__" :
    cs = ContinualScenario(
        dataloader_config={
            'dataloader_cls' : MemoryPoolDataloader,
            'dataloader_kwargs' :{
                'skill_embedding_path' : 'data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',
                'skill_exclude' : None,
                'semantic_flag' : True, 
            }
        },
        evaluator=MMEvaluator(get_task_list_equal_normal(only_normal=False)[:1],
            eval_mode='obs', 
            eval_episodes=1,
        ),
        phase_configures=KITCHEN_MINIMAL_TO_FULL_M,
    )
    dl_kitchen = cs.get_phase_data(0)
    print(dl_kitchen.stacked_data.keys())

    cs_mm = ContinualScenario(
        dataloader_config={
            'dataloader_cls' : MemoryPoolDataloader,
            'dataloader_kwargs' :{
                'skill_embedding_path' : 'data/continual_dataset/evolving_world/mm_lang_embedding.pkl',
                'skill_exclude' : None,
                'semantic_flag' : True, 
            }
        },
        evaluator=MMEvaluator(get_task_list_equal_normal(only_normal=False)[:1],
            eval_mode='obs', 
            eval_episodes=1,
        ),
        phase_configures=MM_EASY_TO_NORMAL_UM,
    )

    dl_mm = cs_mm.get_phase_data(0)
    print(dl_mm.stacked_data.keys())


    k_obs = dl_kitchen.stacked_data['observations'][:,:60]
    k_prev_obs = k_obs.copy()
    k_prev_obs[1:] = k_obs[:-1]

    k_actions = dl_kitchen.stacked_data['actions']
    k_prev_actions = k_actions.copy()
    k_prev_actions[1:] = k_actions[:-1]


    k_actions_dyna = dl_kitchen.stacked_data['actions'].copy() * np.array([1.,0.8,0.5,1.,1. ,0.7,0.8,-1.,1.])
    k_actions_dyna = k_actions_dyna + np.array([0.,0.2,-0.5,0.,0. ,0.1,0.1,0.,0.])
    k_prev_actions_dyna = k_actions_dyna.copy()
    k_prev_actions_dyna[1:] = k_prev_actions_dyna[:-1]

    # zero padd the observation 140
    # k_obs = np.concatenate([k_obs, np.zeros((k_obs.shape[0], 80))], axis=1)
    k_skill = dl_kitchen.stacked_data['skills']

    ## Dynamics embeddings  
    # observations = np.concatenate([ k_prev_obs.copy(), k_prev_actions, k_obs.copy()], axis=-1)
    # observations_dyna = np.concatenate([ k_prev_obs.copy(), k_prev_actions_dyna, k_obs.copy()], axis=-1)
    observations = np.concatenate([  k_prev_actions, k_obs.copy()], axis=-1)
    observations_dyna = np.concatenate([ k_prev_actions_dyna, k_obs.copy()], axis=-1)

    skills = k_skill
    unique_skills = np.unique(skills)

    ## mmworld combination
    mm_obs = dl_mm.stacked_data['observations'][:,:140]
    # mm_obs = dl_mm.stacked_data['observations']
    mm_skill = dl_mm.stacked_data['skills']
    observations = mm_obs
    skills = mm_skill
    unique_skills = np.unique(skills)


    ## fraction dropout
    # observations = np.concatenate([k_actions, k_actions_dyna], axis=0)
    # observations = np.concatenate([observations, observations_dyna], axis=0)

    # B = len(observations)
    # drop_fraction = 0.75  # Fraction of data to drop (e.g., 30%)
    # num_to_drop = int(B * drop_fraction)    
    # indices_to_drop = np.random.choice(B, num_to_drop, replace=False)
    # observations = np.delete(observations, indices_to_drop, axis=0)
    # skills = np.delete(skills, indices_to_drop, axis=0)

    # block_len = len(observations) // 2


    # TSNE start
    tsne = TSNE(n_components=2, random_state=0)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_skills)))
    skill_to_color = {skill: color for skill, color in zip(unique_skills, colors)}
    


    print( "tsne fitting")
    observations_2d = tsne.fit_transform(observations)
    # observations_2d = observations[:,:2]
    print( "fitting done")

    
    # Visualization
    for skill, color in skill_to_color.items():
        idx = skills == skill
        plt.scatter(observations_2d[idx, 0], observations_2d[idx, 1], color=color, label=skill, s=1)
    # plt.scatter(observations_2d[:block_len, 0], observations_2d[:block_len, 1], color=colors[0], label='normal', s=1,alpha=0.15, marker='*')
    # plt.scatter(observations_2d[block_len:, 0], observations_2d[block_len:, 1], color=colors[1], label='dyna', alpha=0.15,s=1)


    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    output_image_path = './data/1207vis/mmskill140.png'  
    plt.savefig(output_image_path, dpi=300) 


    print()

