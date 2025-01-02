import os
root_path = os.environ.get('clus_path')
if root_path is None :
    root_path = os.environ['HOME'] + '/icskill'

def filter_pkl_files_by_skill(directory, include_skill=None, exclude_skill=None):
    """
    Return a list of .pkl files in the given directory that:
    - Contain all the strings in include_skill (if provided and if it's a list) or the include_skill string itself.
    - Do not contain any of the strings in exclude_skill (if provided and if it's a list) or the exclude_skill string itself.
    
    :param directory: The directory to search in.
    :param include_skill: The string or list of strings that files should contain or None.
    :param exclude_skill: The string or list of strings that files should not contain or None.
    :return: A list of matching .pkl filenames.
    """
    # Ensure that skills are in list format
    if include_skill and isinstance(include_skill, str):
        include_skill = [include_skill]
    if exclude_skill and isinstance(exclude_skill, str):
        exclude_skill = [exclude_skill]
    
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter files
    matching_files = [f"{directory}/{f}" for f in files if f.endswith('.pkl') 
                      and (not include_skill or all(skill in f for skill in include_skill))
                      and (not exclude_skill or not any(skill in f for skill in exclude_skill))]
    
    return matching_files

def filter_pkl_files_by_task(directory, exclude_task=None):
    """
    for continual world
    Return a list of .pkl files in the given directory that:
    - Contain all the strings in include_skill (if provided and if it's a list) or the include_skill string itself.
    - Do not contain any of the strings in exclude_skill (if provided and if it's a list) or the exclude_skill string itself.
    
    :param directory: The directory to search in.
    :param include_skill: The string or list of strings that files should contain or None.
    :param exclude_skill: The string or list of strings that files should not contain or None.
    :return: A list of matching .pkl filenames.
    """
    # Ensure that skills are in list format
    if exclude_task and isinstance(exclude_task, str):
        exclude_task = [exclude_task]
    
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter files
    matching_files = [f"{directory}/{f}/expert_traj.pkl" for f in files if f not in exclude_task]
    
    return matching_files

## mmworld config ### 
# dict_keys(['observations', 'actions', 'rewards', 'terminals', 'infos', 'skills', 'skill_done'])
base_mmworld_path = f'{root_path}/data/continual_dataset/evolving_world/raw'
MM_EASY = [
    {
        'data_name' : 'mmeasy',
        'data_paths' : filter_pkl_files_by_skill(f'{root_path}/data/continual_dataset/evolving_world/raw/normal',None ,['handle','lever']),
    },
]

MM_EASY_INCREMETNAL = [
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

MM_NORMAL = [
    {
        'data_name' : 'mmnormal',
        'data_paths' : filter_pkl_files_by_skill(f'{base_mmworld_path}/normal',None ,None),
    },
]

MM_NORMAL_INCREMETNAL = [
    {
        'data_name' : 'puck',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/puck.pkl',
        ],
    },
    {
        'data_name' : 'button',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/button.pkl',
        ],
    },
    {
        'data_name' : 'door',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/door.pkl',
        ],
    },
    {
        'data_name' : 'drawer',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/drawer.pkl',
        ],
    },
    {
        'data_name' : 'handle',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/handle.pkl',
        ],
    },
    {
        'data_name' : 'lever',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/lever.pkl',
        ],
    },
]

MM_EASY_TO_NORMAL_I = [
    {
        'data_name' : 'mmeasy',
        'data_paths' : filter_pkl_files_by_skill(f'{base_mmworld_path}/normal',None ,['handle','lever']),
    },
    {
        'data_name' : 'handle',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/handle.pkl',
        ],
    },
    {
        'data_name' : 'lever',
        'data_paths' : [
            './data/continual_dataset/evolving_world/raw_skill/normal/lever.pkl',
        ],
    },

]

def create_e2m_i(directory, total_phase=96, no_easy_task=False, only_easy_task=False) :
    continual_config = []

    files = os.listdir(directory)
    matching_files = [f"{directory}/{f}" for f in files if f.endswith('.pkl')]
    if no_easy_task :
        matching_files = [f for f in matching_files if 'handle' in f or 'lever' in f]
    elif only_easy_task :
        matching_files = [f for f in matching_files if 'handle' not in f and 'lever' not in f]

    task_per_phase = len(matching_files) // total_phase
    if len(matching_files) % total_phase != 0 :
        task_per_phase += 1

    for i in range(total_phase) :
        task_name = [f.split('/')[-1].split('.')[0]  for f in matching_files[i*task_per_phase:(i+1)*task_per_phase] ]
        task_name = ','.join(task_name)
        # print(task_name)
        continual_config.append({
            'data_name' : task_name,
            'data_paths' : matching_files[i*task_per_phase : (i+1)*task_per_phase],
        })    

    return continual_config

MM_EASY_TO_NORMAL_96 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=96)
MM_EASY_TO_NORMAL_48 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=48)
MM_EASY_TO_NORMAL_24 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=24)
MM_EASY_TO_NORMAL_U72 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=72, no_easy_task=True)
MM_EASY_TO_NORMAL_U36 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=36, no_easy_task=True)
MM_EASY_TO_NORMAL_U24 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=24, no_easy_task=True)
MM_EASY_TO_NORMAL_U12 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=12, no_easy_task=True)
MM_EASY_TO_NORMAL_UM = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=1, no_easy_task=True)

MM_EASY_0 = create_e2m_i(f'{base_mmworld_path}/normal', total_phase=1, only_easy_task=True)


### mmworld config ### 
def create_e2h_i(directory, total_phase=96, no_easy_task=True) :
    continual_config = []

    files = os.listdir(directory)
    matching_files = [f"{directory}/{f}" for f in files if f.endswith('.pkl')]
    if no_easy_task :
        matching_files = [f for f in matching_files if 'handle' in f or 'lever' in f or 'box' in f or 'stick' in f]
    

    set_task = []
    skills = {}
    filtered_file = []
    for i in range(len(matching_files)) :
        if i % 6 == 0 :
            # print(i, matching_files[i].split('/')[-1].split('.')[0])
            filtered_file.append(matching_files[i])
            task = matching_files[i].split('/')[-1].split('.')[0]
            task_aligned = '-'.join(sorted(task.split('-')))
            set_task.append(task_aligned)

            # count the number of skills
            for s in task.split('-') :
                if s not in skills :
                    skills[s] = 0
                skills[s] += 1

    # print(set_task)
    # print(len(set_task))
    # print(set(set_task))
    # print(len(set(set_task)))
    # print(skills)
    matching_files = filtered_file

    task_per_phase = len(matching_files) // total_phase
    if len(matching_files) % total_phase != 0 :
        task_per_phase += 1

    for i in range(total_phase) :
        task_name = [f.split('/')[-1].split('.')[0]  for f in matching_files[i*task_per_phase:(i+1)*task_per_phase] ]
        task_name = ','.join(task_name)
        # print(task_name)
        continual_config.append({
            'data_name' : task_name,
            'data_paths' : matching_files[i*task_per_phase : (i+1)*task_per_phase],
        })    

    return continual_config
MM_EASY_TO_NORMAL_U20 = MM_EASY_TO_NORMAL_U24[:20]
MM_EASY_TO_HARD_U10 = create_e2h_i(
    f'{base_mmworld_path}/hard', 
    total_phase=10, 
    no_easy_task=True
)
MM_EASY_TO_HARD_U20 = create_e2h_i(
    f'{base_mmworld_path}/hard', 
    total_phase=20, 
    no_easy_task=True
)
MM_EASY_TO_HARD_M = create_e2h_i(
    f'{base_mmworld_path}/hard', 
    total_phase=1, 
    no_easy_task=True
)

import random
from itertools import permutations
import numpy as np
def sample_permutations(n):
    # Generate all permutations of '1234'
    all_perms = list(permutations('1234'))

    # Ensure n is not greater than the total number of permutations
    n = min(n, len(all_perms))

    # Randomly select n permutations without replacement
    selected_indices = np.random.choice(len(all_perms), n, replace=False)
    sampled_perms = [all_perms[i] for i in selected_indices]
    return sampled_perms

def create_e2h_hamming2(directory, total_phase=10, multi_task=False) :
    continual_config = []
    files = os.listdir(directory)
    matching_files = [f"{directory}/{f}" for f in files if f.endswith('.pkl')]
    sub_task_pos = [['box','puck'],['handle','drawer'],['lever','button'],['stick','door']]
    hamming2 = ['0000','0011','0100','0001','0010',
                '0101','0110','1000','0111','1001',
                '1010','1100','1011','1101','1110',
                '0010','0111','1011','1000','0001'][:total_phase]
    p_count = 4 if total_phase == 10 else 2
    used_tasks = []
    np.random.seed(777)
    for sub_selection in hamming2 :
        sub_selection = [int(s) for s in sub_selection]
        selected_task = []
        for i in range(len(sub_selection)) :
            selected_task.append(sub_task_pos[i][sub_selection[i]])
        perms = sample_permutations(p_count)
        
        phase_tasks = []
        for ps in perms :
            st = []
            for p in ps :
                st.append(selected_task[int(p)-1])
            task_name = '-'.join(st)
            phase_tasks.append(task_name)
        used_tasks.extend(phase_tasks.copy())
        task_name = ','.join(phase_tasks)

        continual_config.append({
            'data_name' : task_name,
            'data_paths' : [f for f in matching_files if any(s in f for s in phase_tasks)],
        }) 

    if multi_task == True :
        continual_config = []
        task_name = ','.join(used_tasks)
        continual_config.append({
            'data_name' : task_name,
            'data_paths' : [f for f in matching_files if any(s in f for s in used_tasks)],
        })

    return continual_config

# isolated mmworld task
MM_EASY_TO_HARD_H_U10 = create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=10)
MM_EASY_TO_HARD_H_U20 = create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=20)
MM_EASY_TO_HARD_HM_U10 = create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=10, multi_task=True)
MM_EASY_TO_HARD_HM_U20 = create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=20, multi_task=True)


def create_ew_scenario(directory, total_phase=10, multi_task=False) :
    continual_config = []
    files = os.listdir(directory)
    matching_files = [f"{directory}/{f}" for f in files if f.endswith('.pkl')]
    sub_task_pos = [['puck','box'],['handle','drawer'],['lever','button'],['stick','door']]
    hamming2 = ['0000','0011','0100','0001','0010',
                '0101','0110','1011','0111','1001',
                '1010','1100','1000','1101','1110',
                '0010','0111','1011','1000','0001'][:total_phase]
    
    if total_phase == 10 : 
        hamming2.extend( hamming2.copy())
    
    p_count =1 # conut of tasks
    used_tasks = []
    np.random.seed(777)
    for sub_selection in hamming2 :
        sub_selection = [int(s) for s in sub_selection]
        selected_task = []
        for i in range(len(sub_selection)) :
            selected_task.append(sub_task_pos[i][sub_selection[i]])
        perms = sample_permutations(p_count)
        
        phase_tasks = []
        for ps in perms :
            st = []
            for p in ps :
                st.append(selected_task[int(p)-1])
            task_name = '-'.join(st)
            phase_tasks.append(task_name)
        used_tasks.extend(phase_tasks.copy())
        task_name = ','.join(phase_tasks)

        continual_config.append({
            'data_name' : task_name,
            'data_paths' : [f for f in matching_files if any(s in f for s in phase_tasks)],
        }) 

    if multi_task == True :
        continual_config = []
        task_name = ','.join(used_tasks)
        continual_config.append({
            'data_name' : task_name,
            'data_paths' : [f for f in matching_files if any(s in f for s in used_tasks)],
        })

    return continual_config


def create_ew_scenario_ez(directory, total_phase=20, stype='complete', unseen=None) :
    # unseen : ind / unseen
    continual_config = []
    continual_config_r = []

    files = os.listdir(directory)
    matching_files = [f"{directory}/{f}" for f in files if f.endswith('.pkl')]

    if stype == 'semi' : 
        total_phase = 10     
    
    sub_task_pos = [['puck','box'],['drawer','handle'],['button','lever'],['door','stick']]
    hamming2 = ['1100','0011','0110','1001','1100',
                '0011','0110','1001','1100','0011',

                '0110','1001','1100','0011','0110',
                '1001','1100','0011','0110','1001',][:total_phase]
    
    base_env = [
        '1100', '0011', '0110', '1001'
    ]
    np.random.seed(777)
    base_env_seq_dict = {
        env : sample_permutations(24) for env in base_env
    }

    # ## process unseen sampling
    # if unseen == 'ind' :
    #     for k in base_env_seq_dict.keys() :
    #         base_env_seq_dict[k] = base_env_seq_dict[k].reverse()

    base_env_appear_count = {
        env : 0 for env in base_env
    }
    task_per_phase = 1
    for sub_selection_id in hamming2 :
        # for int id to actual task
        sub_selection = [int(s) for s in sub_selection_id]
        selected_task = []
        for i in range(len(sub_selection)) :
            selected_task.append(sub_task_pos[i][sub_selection[i]])
        
        phase_tasks = []
        for t_id in range(task_per_phase) : 
            task_appear = base_env_appear_count[sub_selection_id]
            task_seq = base_env_seq_dict[sub_selection_id][task_appear]
            st = []
            for s in task_seq :
                st.append(selected_task[int(s)-1])
            task_name = '-'.join(st)
            phase_tasks.append(task_name)

            base_env_appear_count[sub_selection_id] += 1


        continual_config.append({
            'data_name' : ','.join(phase_tasks),
            'data_paths' : [f for f in matching_files if any(s in f for s in phase_tasks)],
        }) 
        continual_config_r.append({
            'data_name' : ','.join(phase_tasks),
            'data_paths' : [f for f in matching_files if any(s in f for s in phase_tasks)],
        })

    ### indistribution unseen processing 
    if unseen == 'ind' or unseen == 'ius':
        unseen_base_env = [
            '1100', '0011', '0110', '1001'
        ]
        np.random.seed(777)
        unseen_base_env_seq_dict = {
            env : sample_permutations(24) for env in unseen_base_env
        }
        unseen_base_env_appear_count = {
            env : 0 for env in unseen_base_env
        }
    elif unseen == 'unseen' : 
        unseen_base_env = [
        '0010', '0100', '0111', '1000',
        ]
        np.random.seed(777)
        unseen_base_env_seq_dict = {
            env : sample_permutations(24) for env in unseen_base_env
        }
        unseen_base_env_appear_count = {
            env : 0 for env in unseen_base_env
        }
    if unseen is not None :
        unseen_task_buffers = []
        for us_id, sub_selection_id  in enumerate(hamming2) :
            sub_selection_id = unseen_base_env[us_id%4]
            # for int id to actual task
            sub_selection = [int(s) for s in sub_selection_id]
            selected_task = []
            for i in range(len(sub_selection)) :
                selected_task.append(sub_task_pos[i][sub_selection[i]])
            phase_tasks = []
            for t_id in range(task_per_phase) : 
                task_appear = unseen_base_env_appear_count[sub_selection_id] + 1
                # print(len(unseen_base_env_seq_dict[sub_selection_id]))
                # find unseen in reverse
                task_seq = unseen_base_env_seq_dict[sub_selection_id][-task_appear]
                st = []
                for s in task_seq :
                    st.append(selected_task[int(s)-1])
                task_name = '-'.join(st)
                unseen_task_buffers.append(task_name)   
                unseen_base_env_appear_count[sub_selection_id] += 1

            if us_id%4 == 3 :
                original_task_name = continual_config[us_id]['data_name']
                unseen_task_name = ','.join(unseen_task_buffers)
                continual_config[us_id]['data_name'] = ','.join([original_task_name, unseen_task_name])
                unseen_task_buffers = []

    if unseen == 'ius' :
        unseen_base_env = [
        '0010', '0100', '0111', '1000',
        ]
        np.random.seed(777)
        unseen_base_env_seq_dict = {
            env : sample_permutations(24) for env in unseen_base_env
        }
        unseen_base_env_appear_count = {
            env : 0 for env in unseen_base_env
        }
        unseen_task_buffers = []
        for us_id, sub_selection_id  in enumerate(hamming2) :
            sub_selection_id = unseen_base_env[us_id%4]
            # for int id to actual task
            sub_selection = [int(s) for s in sub_selection_id]
            selected_task = []
            for i in range(len(sub_selection)) :
                selected_task.append(sub_task_pos[i][sub_selection[i]])
            phase_tasks = []
            for t_id in range(task_per_phase) : 
                task_appear = unseen_base_env_appear_count[sub_selection_id]+1
                # find unseen in reverse
                task_seq = unseen_base_env_seq_dict[sub_selection_id][-task_appear]
                st = []
                for s in task_seq :
                    st.append(selected_task[int(s)-1])
                task_name = '-'.join(st)
                unseen_task_buffers.append(task_name)   
                unseen_base_env_appear_count[sub_selection_id] += 1

            if us_id%4 == 3 :
                original_task_name = continual_config[us_id]['data_name']
                unseen_task_name = ','.join(unseen_task_buffers)
                continual_config[us_id]['data_name'] = ','.join([original_task_name, unseen_task_name])
                unseen_task_buffers = []


    
    if total_phase == 10 :
        continual_config.extend(continual_config_r.copy())

    # remove skill selection


    return continual_config

E2H_U10_RMSKILLS = [ # p3 and p4 is single
    'box', 'handle', 'lever', 'stick', 'button',
    'door', 'drawer', 'puck', 'box', 'handle',
    'lever', 'button', 'door', 'box', 'handle',
    'drawer', 'button', 'stick', 'door', 'lever',
]

# for fixed version 
E2H_U10_RMSKILLS2 = [
    'lever', 'button', 'door', 'puck', 'handle',
    'drawer', 'button', 'stick', 'door', 'lever',
    'puck', 'handle', 'lever', 'stick', 'button',
    'door', 'drawer', 'puckbox', 'puck', 'handle',
]

EW_RMSKILL_INCOMP = [
    'door', 'puck', 'lever', 'box', 
    'button', 'drawer', 'handle', 'button', 
    'handle', 'lever', 'door', 'drawer', 
    'box', 'stick', 'lever', 'button', 
    'button', 'drawer', 'puck', 'stick',
]

EW_RMSKILL_SECOM = [
    'door', 'puck', 'lever', 'box', 
    'button', 'drawer', 'handle', 'button', 
    'handle', 'lever',  
    'box', 'stick', 'door', 'button', 
    'handle', 'lever', 'puck', 'stick',
    'button', 'drawer',  
]

MW_COMPLETE = create_ew_scenario_ez(f'{base_mmworld_path}/hard', total_phase=20)
MW_SEMI_COMPLETE = create_ew_scenario_ez(f'{base_mmworld_path}/hard', total_phase=10)
for i , data in enumerate(MW_SEMI_COMPLETE) :
    data['skill_exclude'] = EW_RMSKILL_SECOM[i]
    if EW_RMSKILL_SECOM[i] not in data['data_name'] :
        print('error-semi', i, EW_RMSKILL_SECOM[i], data['data_name'])
    if i > 10 :
        if MW_SEMI_COMPLETE[i-10]['skill_exclude'] == data['skill_exclude'] :
            print( 'error-semi', i, EW_RMSKILL_SECOM[i], data['data_name'])

MW_INCOMPLETE = create_ew_scenario_ez(f'{base_mmworld_path}/hard', total_phase=20)
for i , data in enumerate(MW_INCOMPLETE) :
    data['skill_exclude'] = EW_RMSKILL_INCOMP[i]
    if EW_RMSKILL_INCOMP[i] not in data['data_name'] :
        print('error-incom', i, EW_RMSKILL_INCOMP[i], data['data_name'])

# check duplicated task 
tasks = []
for i , data in enumerate(MW_INCOMPLETE) :
    if data['data_name'] in tasks : 
        print('error', i, data['data_name'])
    else : 
        tasks.append(data['data_name'])

for i , data in enumerate(MW_INCOMPLETE) :
    data['skill_exclude'] = EW_RMSKILL_INCOMP[i]
    if EW_RMSKILL_INCOMP[i] not in data['data_name'] :
        print('error', i, EW_RMSKILL_INCOMP[i], data['data_name'])


MM_COMPLETE_IND = create_ew_scenario_ez(f'{base_mmworld_path}/hard', total_phase=20, unseen='ind')
MM_COMPLETE_UNSEEN = create_ew_scenario_ez(f'{base_mmworld_path}/hard', total_phase=20, unseen='unseen')
MM_COMPLETE_IUS = create_ew_scenario_ez(f'{base_mmworld_path}/hard', total_phase=20, unseen='ius')

UNSEEN_IND =  ['handle-box-door-button', 'drawer-puck-stick-lever', 'door-handle-lever-puck', 'stick-drawer-box-button', 'button-handle-door-box', 'lever-puck-stick-drawer', 'lever-handle-puck-door', 'button-stick-drawer-box', 'handle-box-button-door', 'puck-stick-drawer-lever', 'door-lever-handle-puck', 'drawer-button-box-stick', 'button-door-handle-box', 'drawer-puck-lever-stick', 'puck-door-lever-handle', 'drawer-button-stick-box', 'door-button-handle-box', 'puck-lever-drawer-stick', 'puck-handle-door-lever', 'button-box-drawer-stick']
UNSEEN_UNSEEN= ['drawer-puck-door-lever', 'handle-puck-door-button', 'stick-handle-lever-puck', 'door-drawer-box-button', 'lever-drawer-door-puck', 'button-puck-door-handle', 'lever-handle-puck-stick', 'button-door-drawer-box', 'drawer-puck-lever-door', 'puck-door-handle-button', 'stick-lever-handle-puck', 'drawer-button-box-door', 'lever-door-drawer-puck', 'handle-puck-button-door', 'puck-stick-lever-handle', 'drawer-button-door-box', 'door-lever-drawer-puck', 'puck-button-handle-door', 'puck-handle-stick-lever', 'button-box-drawer-door']

### sparse config ###
MM_EASY_TO_HARD_HS_U20 =  create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=10)
MM_EASY_TO_HARD_HS_U20.extend( create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=10))
for i, data in enumerate(MM_EASY_TO_HARD_HS_U20) :
    data['skill_exclude'] = E2H_U10_RMSKILLS[i]

MM_EASY_TO_HARD_HS_U20_2 =  create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=10)
MM_EASY_TO_HARD_HS_U20_2.extend( create_e2h_hamming2(f'{base_mmworld_path}/hard', total_phase=10))
for i, data in enumerate(MM_EASY_TO_HARD_HS_U20_2) :
    data['skill_exclude'] = E2H_U10_RMSKILLS2[i]






### kitchen config ###
base_kitchen_path = f'{root_path}/data/continual_dataset/evolving_kitchen/raw_skill'
base_kitchen_path_raw = f'{root_path}/data/continual_dataset/evolving_kitchen/raw'
base_kitchen_path_minimal = f'{root_path}/data/continual_dataset/evolving_kitchen/minimal'


# minimal : k b t l
# standard : m k b t l 
# full : m k b t l h s
KITCHEN_MINIMAL= [
    {
        'data_name' : 'minimal',
        'data_paths' :  filter_pkl_files_by_skill(f'{base_kitchen_path_minimal}'),
    },
]

## Action Dynamics modified configures ##
## scale must be larger than 0.5
## if scale is smaller than 0.5 action is hard to predict by model

import numpy as np

def create_m2f_i(directory, dynamics=None , total_phase=24) : 
    continual_config = []
    matching_files = filter_pkl_files_by_skill(directory)
    task_per_phase = len(matching_files) // total_phase
    
    if len(matching_files) % total_phase != 0 :
        task_per_phase += 1

    for i in range(total_phase) :
        task_name = [f.split('/')[-1].split('.')[0]  for f in matching_files[i*task_per_phase:(i+1)*task_per_phase] ]
        task_name = ','.join(task_name)
        # print(task_name)
        if dynamics is not None :
            j = i % len(dynamics)
            continual_config.append({
                'data_name' : task_name,
                'data_paths' : matching_files[i*task_per_phase : (i+1)*task_per_phase],
                'dynamics' : dynamics[j],
            })
        else :
            continual_config.append({
                'data_name' : task_name,
                'data_paths' : matching_files[i*task_per_phase : (i+1)*task_per_phase],
            })    

    return continual_config

## TODO Desinged for 24 phases
KITCHEN_MINIMAL_TO_FULL_24 = create_m2f_i(f'{base_kitchen_path_raw}', total_phase=24)
KITCHEN_MINIMAL_TO_FULL_12 = create_m2f_i(f'{base_kitchen_path_raw}', total_phase=12)
KITCHEN_MINIMAL_TO_FULL_6 = create_m2f_i(f'{base_kitchen_path_raw}', total_phase=6)
KITCHEN_MINIMAL_TO_FULL_M = create_m2f_i(f'{base_kitchen_path_raw}', total_phase=1)

KITCHEN_MINIMAL = create_m2f_i(f'{base_kitchen_path_minimal}', total_phase=1)

## domain modified configure 
def domain_add( config, domain ) :
    domain_count = len(domain)
    return_config = []
    for i, c in enumerate(config) :
        return_config.append(c.copy())
        return_config[i]['domain'] = domain[i % domain_count].copy()
    return return_config


KITCHEN_INC = [
    {
        'data_name' : 'microwave',
        'data_paths' : [
            f'{base_kitchen_path}/microwave.pkl',
        ],
    },
    {
        'data_name' : 'kettle',
        'data_paths' : [
            f'{base_kitchen_path}/kettle.pkl',
        ],
    },
    {
        'data_name' : 'bottom burner',
        'data_paths' : [
            f'{base_kitchen_path}/bottom burner.pkl',
        ],
    },
    {
        'data_name' : 'top burner',
        'data_paths' : [
            f'{base_kitchen_path}/top burner.pkl',
        ],
    },
    {
        'data_name' : 'light switch',
        'data_paths' : [
            f'{base_kitchen_path}/light switch.pkl',
        ],
    },
    {
        'data_name' : 'hinge cabinet',
        'data_paths' : [
            f'{base_kitchen_path}/hinge cabinet.pkl',
        ],
    },
    {
        'data_name' : 'slide cabinet',
        'data_paths' : [
            f'{base_kitchen_path}/slide cabinet.pkl',
        ],
    },
]  



######### 0504 Kitchen Cofigures #############
def evolving_kitchen(type=None, unseen=None) :
    seq_list =[
        'mktl','kbts','mbts','kbls','mkls',
        'kbth','mkth','mksh','klsh','mkbh',
        'kbsh','kblh','mtlh','mkbs','mklh',
        'mbtl','ktls','mbth','mbsh','mbls',
    ]

    semi_exclude = [ # 10 + 10 
        'k', 'b', 't', 'l', 's',
        'h', 'm', 's', 'h', 'b',

        'm', 'k', 'b', 's', 'k',
        't', 'h', 'k', 's', 'm',
    ]
    incomplete_exclude = [
        'k', 'b', 't', 'l', 's',
        'h', 'm', 's', 'h', 'b',

        'k', 'b', 'h', 'm', 'l',
        'b', 'k', 'm', 's', 'b',
    ]

    if type == 'complete' or type == 'incomplete' :
        seq_list = seq_list[:20]
    elif type == 'complete10' :
        seq_list = seq_list[:10] + seq_list[:10].copy()
    elif type == 'semi' :
        seq_list = seq_list[:10] + seq_list[:10].copy()

    continual_config = []
    initial_dict ={
            'm' : 'microwave',
            'k' : 'kettle',
            'b' : 'bottom burner',
            't' : 'top burner',
            'l' : 'light switch',
            'h' : 'hinge cabinet',
            's' : 'slide cabinet',
        }
    def initial_to_name(initial:str) :
        return '-' . join([initial_dict[i] for i in initial])
    
    kitchen_paths = base_kitchen_path_raw
    matching_files = filter_pkl_files_by_skill(kitchen_paths)

    for seq in seq_list :
        data_name = initial_to_name(seq)
        data_paths = [f for f in matching_files if data_name in f]
        continual_config.append({
            'data_name' : data_name,
            'data_paths' : data_paths,
        })

    if type == 'semi' :
        for pid, config in enumerate(continual_config ):
            config['skill_exclude'] = initial_dict[semi_exclude[pid]]
        
    elif type == 'incomplete' :
        for pid, config in enumerate(continual_config ):
            config['skill_exclude'] = initial_dict[incomplete_exclude[pid]]


    if unseen is not None :
        unseen_list = [
            ['mkts', 'mkbt'], 
            ['mkbl', 'kbtl'],
            ['ktsh', 'mtsh'], 
            ['mtls', 'ktlh'],
        ]
        for us_id, config in enumerate(continual_config) :
            if us_id % 5 == 4 :
                unseen_group = us_id // 5
                unseen_tasks = [ initial_to_name(seq) for seq in unseen_list[unseen_group]]
                # add origianl tasks to front 
                unseen_tasks = [config['data_name']] + unseen_tasks
                continual_config[us_id]['data_name'] = ','.join(unseen_tasks)

    return continual_config


EK_COMPLETE = evolving_kitchen('complete')
EK_COMPLETE10 = evolving_kitchen('complete10')
EK_SEMI = evolving_kitchen('semi')
EK_INCOMPLETE = evolving_kitchen('incomplete')

EK_COMP_base = evolving_kitchen('complete')
pass_tasks = [3,7,12,16]
EK_COMP_RET = []
for i, d in enumerate(EK_COMP_base) :
    if i in pass_tasks :
        continue
    EK_COMP_RET.append(d)

EK_INCOMP_base = evolving_kitchen('incomplete')
pass_tasks = [3,7,12,16]
EK_INCOMP_RET = []
for i, d in enumerate(EK_INCOMP_base) :
    if i in pass_tasks :
        continue

    EK_INCOMP_RET.append(d)

try:
    import copy
    unlearn_tasks = [3,7,12,16]
    unlearn_stages = [4,9,14,19]
    UEK_COMPLETE_base = evolving_kitchen('complete')
    UEK_COMPLETE = []
    unlearn_stage_count = 0
    for i, d in enumerate(UEK_COMPLETE_base) :
        d['mode'] = 'learn'
        UEK_COMPLETE.append(d)
        if i in unlearn_stages :
            unlearn_stage = copy.deepcopy(UEK_COMPLETE_base[unlearn_tasks[unlearn_stage_count]])
            unlearn_stage['mode'] = 'unlearn'
            UEK_COMPLETE.append(unlearn_stage)
            unlearn_stage_count += 1
    
    UEK_INCOMPLETE_base = evolving_kitchen('incomplete')
    UEK_INCOMPLETE = []
    unlearn_stage_count = 0
    for i, d in enumerate(UEK_INCOMPLETE_base) :
        d['mode'] = 'learn'
        UEK_INCOMPLETE.append(d)
        if i in unlearn_stages :
            unlearn_stage = copy.deepcopy(UEK_INCOMPLETE_base[unlearn_tasks[unlearn_stage_count]])
            unlearn_stage['mode'] = 'unlearn'
            UEK_INCOMPLETE.append(unlearn_stage)
            unlearn_stage_count += 1
            
    # for stg in UEK_INCOMPLETE :
    #     print(stg['data_name'], stg['mode'])
    #     if stg['mode'] == 'unlearn' :
    #         print("-------------")
except : 
    pass
# for i, d in enumerate(EK_SEMI):
#     print(i, d['data_name'], '!!!', d['skill_exclude'])


EK_COMPLETE_IND = evolving_kitchen('complete', unseen='ind')
unseen_list = [
            ['mkts', 'mkbt'], 
            ['mkbl', 'kbtl'],
            ['ktsh', 'mtsh'], 
            ['mtls', 'ktlh'],
        ]

# KITCHEN_UNSEEN= [
#     'microwave-kettle-top burner-slide cabinet',
#     'microwave-kettle-bottom burner-top burner',
#     'microwave-kettle-bottom burner-light switch',
#     'kettle-bottom burner-top burner-light switch',
#     'kettle-top burner-slide cabinet-hinge cabinet',
#     'microwave-top burner-slide cabinet-hinge cabinet',
#     'microwave-top burner-light switch-slide cabinet',
#     'kettle-top burner-light switch-hinge cabinet',
# ]


KITCHEN_UNSEEN= [
    'microwave-kettle-top-burner-slide-cabinet',
    'microwave-kettle-bottom-burner-top-burner',
    'microwave-kettle-bottom-burner-light-switch',
    'kettle-bottom-burner-top-burner-light-switch',
    'kettle-top-burner-slide-cabinet-hinge-cabinet',
    'microwave-top-burner-slide-cabinet-hinge-cabinet',
    'microwave-top-burner-light-switch-slide-cabinet',
    'kettle-top-burner-light-switch-hinge-cabinet',
]

for t in EK_COMPLETE_IND :
    # print(t['data_name'])
    pass


if __name__ == '__main__' :
    evolving_kitchen('complete20')

possible_evaluation = [
    'mtlh',
    # 'mlsh',
    'mktl',
    'mkth',
    'mksh',

    'mkls',
    'mklh',
    'mkbs',
    'mkbh',
    'mbts',

    'mbtl',
    'mbth',
    'mbsh',
    'mbls',
    'ktls',
    
    'klsh',
    'kbts',
    #'kbtl',
    'kbth',
    'kbsh',
    
    'kbls',
    'kblh',
    #'btsh',
    #'btls',
]

minimal = [
    'kbtl',
    # partial
    'kl',
    'ktl',
    'kbt',
    'kbl',
    'bt',
    'btls', 
]