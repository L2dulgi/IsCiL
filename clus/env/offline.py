
import pickle
import random as pyrand
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

class BaseDataloader() :
    def __init__(self,
        data_paths=[
            'data/kitchen/kitchen_skill_appended.pkl', 
        ], 
        skill_embedding_path='data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',

        semantic_flag=False,
        skill_exclude:str=None, 
        exclude_mode:str='skill', # ['episode', 'skill']
        traj_len=1,

        few_shot_len=None,
        few_random_flag=False,
        ) :
        with open( skill_embedding_path , 'rb' ) as f :
            skill_embedding = pickle.load(f)
        self.skill_embedding = skill_embedding
        self.semantic_flag = semantic_flag

        self.skill_exclude = skill_exclude
        self.exclude_mode = exclude_mode
        self.traj_len = traj_len

        self.data_paths = data_paths
        self.stacked_data=None
        self.load_data()
        self.process_data()
        self.process_oracle_context()

        if few_shot_len != None :
            if few_random_flag is True :
                self.cut_data_rand(few_shot_len)
            else :
                self.cut_data(few_shot_len)
    
    def cut_data_rand(self, few_shot_len) :
        '''random few shot data generation'''
        np.random.seed(777)
        indicies = np.random.randint(0, self.dataset_size, few_shot_len*10)
        self.stacked_data = {
            key : self.stacked_data[key][indicies] for key in self.stacked_data.keys()
        }
        self.dataset_size = len(self.stacked_data['observations'])
        print(f"dataset size : {self.dataset_size}")

    def cut_data(self, few_shot_len) :
        '''few trajectory data generation'''
        episode_count = len(self.stacked_data['terminals'].nonzero()[0])
        if episode_count < few_shot_len :
            print(f"[dataloader]episode count : {episode_count} is less than few_shot_len : {few_shot_len} not truncated")
            return
        print(f"[dataloader] episode count : {episode_count} is more than few_shot_len : {few_shot_len} truncated")

        few_episode_id = self.stacked_data['terminals'].nonzero()[0][few_shot_len] + 1
        self.stacked_data = {
            key : self.stacked_data[key][:few_episode_id] for key in self.stacked_data.keys()
        }
        self.dataset_size = len(self.stacked_data['observations'])
        print(f"dataset size : {self.dataset_size}")
    
    def process_oracle_context(self) :
        pass
        ## process by trajectory Fixed version
        # self.stacked_data['daco_context'] = np.zeros_like(self.stacked_data['observations'])
        # for i in range(len(self.data_partition)-1) :
        #     partition_start = 0 if i == 0 else self.data_partition[i-1]
        #     partition_end = self.data_partition[i]
        #     partition_terminals = self.stacked_data['terminals'][partition_start:partition_end]
        #     partition_obs = self.stacked_data['observations'][partition_start:partition_end]
        #     ep_done = np.where(np.array(partition_terminals) == 1)[0][0]
        #     first_traj = np.array(partition_obs)[:ep_done]
        #     context = np.mean(first_traj, axis=0)
        #     self.stacked_data['daco_context'][partition_start:partition_end] = context

    def load_data(self) :
        data = None
        self.data_partition = []
        for data_path in self.data_paths :
            with open( data_path , 'rb' ) as f :
                loaded_data = pickle.load(f)
            if data is None :
                data = loaded_data
            else :
                for key in data.keys() :
                    if type(data[key]) == list :
                        data[key].extend(loaded_data[key])
                    elif type(data[key]) == np.ndarray :
                        data[key] = np.concatenate([data[key], loaded_data[key]], axis=0)
                    else : 
                        raise NotImplementedError
            self.data_partition.append(len(data['observations']))
        for key in data.keys() :
            if type(data[key]) == list :
                data[key] = np.array(data[key])
        self.data_buffer = data

    def merge(self, dataloader) :
        for key in self.stacked_data.keys() :
            if type(self.stacked_data[key]) == list :
                self.stacked_data[key].extend(dataloader.stacked_data[key])
            elif type(self.stacked_data[key]) == np.ndarray :
                self.stacked_data[key] = np.concatenate([self.stacked_data[key], dataloader.stacked_data[key]], axis=0)
            else : 
                raise NotImplementedError
        self.dataset_size = len(self.stacked_data['observations'])

    def process_data(self) :
        print('skill embedding size : ', self.skill_embedding.keys())
        if self.skill_exclude is not None : # NOTE tobe deprecated if data is loaded by pkl 
            if self.exclude_mode == 'skill' :
                for i, skill in enumerate(self.data_buffer['skills']) :
                    if skill not in self.skill_exclude :
                        for key in self.data_buffer.keys() :
                            self.stacked_data[key].append( self.data_buffer[key][i] )
            elif self.exclude_mode == 'episode' :
                ep_start_idx = 0
                for i, terminal in enumerate(self.data_buffer['terminals']) :
                    if terminal is True :
                        if self.skill_exclude not in set(self.data_buffer['skills'][ep_start_idx:i]) :
                            for key in self.data_buffer.keys() :
                                self.stacked_data[key].extend( self.data_buffer[key][ep_start_idx:i] )
                        ep_start_idx = i+1
        else : 
            self.stacked_data = self.data_buffer

        if self.semantic_flag is True :
            B = len(self.stacked_data['observations'])
            F = self.skill_embedding[self.stacked_data['skills'][0]].shape[-1] + self.stacked_data['observations'][0].shape[-1]
            augmented_observations = np.zeros((B,F))
            for i, obs in enumerate(self.stacked_data['observations']) :
                augmented_observations[i] = np.concatenate(
                    [obs, self.skill_embedding[self.stacked_data['skills'][i]]]
                )
            self.stacked_data['observations'] = augmented_observations
        else : 
            pass
            # for i, obs in enumerate(self.stacked_data['observations']) :
            #     self.stacked_data['skills'][i] = self.skill_embedding[self.stacked_data['skills'][i]] 

        self.stacked_data = {
            key : np.array(self.stacked_data[key]) for key in self.stacked_data.keys()
        }

        self.terminal_indicies = self.stacked_data['terminals'].nonzero()[0]
        self.traj_indicies = []
        
        ## seq loader processing part
        if self.traj_len is not 1 :
            print(f"[dataloader] sequence data processing on traj_len{self.traj_len}")
            prev_terminal = -1
            for curr_terminal in self.terminal_indicies :
                # append initial to prev_terminal + 1  terminal - traj_len + 1
                append_indicies = np.arange(prev_terminal+1, curr_terminal-self.traj_len+1)
                self.traj_indicies.append(append_indicies)
                prev_terminal = curr_terminal
            self.traj_indicies = np.concatenate(self.traj_indicies)
        
        self.dataset_size = len(self.stacked_data['observations'])

        # episode boundary processing 
        self.episode_boundary = None # (D, 2) first = start, second = end  
        self.stacked_data['daco_context'] = np.zeros_like(self.stacked_data['observations'])

        terminals = np.array(self.stacked_data['terminals'])
        terminal_pos = np.where(terminals)[0]
        self.episode_boundary = np.zeros((len(self.stacked_data['observations']), 2), dtype=np.int32)
        terminal_pos = np.insert(terminal_pos, 0, -1)
        for i in range(1, len(terminal_pos)):
            self.episode_boundary[terminal_pos[i-1]+1:terminal_pos[i]+1, 0] = terminal_pos[i-1] + 1
            self.episode_boundary[terminal_pos[i-1]+1:terminal_pos[i]+1, 1] = terminal_pos[i]
            mean_traj = np.mean(self.stacked_data['observations'][terminal_pos[i-1]+1:terminal_pos[i]+1], axis=0)
            self.stacked_data['daco_context'][terminal_pos[i-1]+1:terminal_pos[i]+1] = mean_traj.copy()
        self.stacked_data['episode_boundary'] = self.episode_boundary

        # process the oracle context
        

        # process if actions are in (-1,1) boundary
        if np.max(self.stacked_data['actions']) > 1.0 or np.min(self.stacked_data['actions']) < -1.0 :
            print("[dataloader] actions are not in (-1,1) boundary")
            print(f"[dataloader] actions cliped from {np.min(self.stacked_data['actions'])} ~ {np.max(self.stacked_data['actions'])}")
            self.stacked_data['actions'] = np.clip(self.stacked_data['actions'], -1.0, 1.0)

    def get_rand_batch(self, batch_size=None) :
        if batch_size == -1 :
            return self.stacked_data
        indicies = np.random.choice( self.dataset_size , batch_size )
        batch = {
            key : self.stacked_data[key][indicies] for key in self.stacked_data.keys()
        }
        return batch
    
    def get_all_batch(self, batch_size=None) :
        indicies = np.arange( self.dataset_size )
        np.random.shuffle(indicies)
        for i in range(0, self.dataset_size, batch_size) :
            batch = {
                key : self.stacked_data[key][indicies[i:i+batch_size]] for key in self.stacked_data.keys()
            }
            yield batch
    
    ### sequential batch method ###
    def get_all_seq_batch(self, batch_size=None) :
        indicies = self.traj_indicies.copy()
        np.random.shuffle(indicies)
        
        for i in range(0, len(indicies), batch_size) :
            bid = indicies[i:i+batch_size]
            bidj = bid + self.traj_len
            for j in range(self.traj_len) :
                if j == 0 :
                    batch = {
                        key : [] for key in self.stacked_data.keys()
                    }
                for key in self.stacked_data.keys() :
                    batch[key].append(self.stacked_data[key][bid+j])

            # concatnate batch by axis 1 or hstack (S, B, F) -> (B, S, F)
            batch = {
                key : np.stack(batch[key], axis=1) for key in batch.keys()
            }

            yield batch

    # temporal sampling method
    def get_kmeans_batch(self, batch_size=None) :
        print("kmeans batch sampling")
        print("balanced kmeans batch sampling")
        def balanced_kmeans(X, K):
            n_samples, _ = X.shape
            initial_clusters = np.random.choice(K, n_samples)
            max_cluster_size = n_samples // K
            
            for i in tqdm(range(1000)):
                centroids = np.array([X[initial_clusters == k].mean(axis=0) for k in range(K)])
                distances = pairwise_distances(X, centroids)
                new_clusters = np.argmin(distances, axis=1)
                
                for i in range(n_samples):
                    cluster_counts = np.bincount(new_clusters)
                    while cluster_counts[new_clusters[i]] > max_cluster_size:
                        distances[i, new_clusters[i]] = float('inf')
                        new_clusters[i] = np.argmin(distances[i])
                        cluster_counts = np.bincount(new_clusters)
                
                if np.all(new_clusters == initial_clusters):
                    break
                
                initial_clusters = new_clusters
            
            return new_clusters
        data = self.stacked_data['actions'].copy()
        n_clusters = batch_size

        # kmeans = KMeans(n_clusters=n_clusters).fit(data)
        # cluster_centers = kmeans.cluster_centers_

        labels = balanced_kmeans(data, n_clusters)
        cluster_centers = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        nearest_sample_indices = []
        for center in cluster_centers:
            distances = np.linalg.norm(data - center, axis=1)
            
            nearest_sample_index = np.argmin(distances)
            nearest_sample_indices.append(nearest_sample_index)
        
        nearest_sample_index = np.array(nearest_sample_indices)

        batch = {
            key : self.stacked_data[key][nearest_sample_index] for key in self.stacked_data.keys()
        }
        return batch

class DynamicsDataloader(BaseDataloader) :
    def __init__(
            self,
            domain_info=None,
            **kwargs
        ) -> None:
        self.domain_info_example = { 
            'scale' : np.array([1.,1.,1.,1.,1., 1.,1.,1.,1.]), # action scale
            'shift' : np.array([0.,0.,0.,0.,0., 0.,0.,0.,0.]), # shifted distribution
        }
        
        self.domain_info = domain_info
        super().__init__(**kwargs)

    def load_data(self):
        super().load_data()
        print(f"[dynamics added to data]")
        print(f"[dynamics info] {self.domain_info}")

        if self.domain_info is not None :
            apply_len = self.domain_info['scale'].shape[0]
            self.data_buffer['observations'][:,:apply_len] = self.domain_info['scale'] * self.data_buffer['observations'][:,:apply_len].copy() 
            self.data_buffer['observations'][:,:apply_len] = self.domain_info['shift'] + self.data_buffer['observations'][:,:apply_len].copy()
        self.stacked_data = self.data_buffer


class AppendedStateDataloader(BaseDataloader) :
    def __init__(
            self,
            max_obs_dim=200,
            max_action_dim=10,
            **kwargs
        ) -> None:
        # TODO change it to integrated state processor
            # requirs max_obs_dim, max_action_dim
            # function for process input output by 'env_id'

        self.max_obs_dim = max_obs_dim
        self.max_action_dim = max_action_dim
        super().__init__(**kwargs)

    def load_data(self):
        super().load_data()
        if self.max_obs_dim < self.data_buffer['observations'][0].shape[-1] :
            print(f'[AppendedStateDataloader] observation dimension is appended to {self.obs_dim} \n data has {self.data_buffer["observations"].shape[-1]}')
            append_dim = self.max_obs_dim - self.data_buffer['observations'].shape[-1]
            self.data_buffer['observations'] = np.concatenate(
                [self.data_buffer['observations'], np.zeros((len(self.data_buffer['observations']), append_dim))], axis=-1
            )

import numpy as np
class MemoryPoolDataloader(DynamicsDataloader):
    '''
    # DataLoader for pool_key based sampling
    '''
    def __init__(
            self, 
            skill_exclude = None,
            **kwargs,
        ) -> None:
        print("[Sparse mode MemoryPoolDataloader]")
        super().__init__(**kwargs)
        self.skill_exclude=skill_exclude
        print(f"[Sparse mode MemoryPoolDataloader] remove skill : {self.skill_exclude}")
        print(f"[Sparse mode MemoryPoolDataloader] data size : {len(self.stacked_data['observations'])}")
        for key in self.stacked_data.keys() :
            indicies = np.where(self.stacked_data['skills'] != self.skill_exclude)[0]
            self.stacked_data[key] = self.stacked_data[key][indicies]
        print(f"[Sparse mode MemoryPoolDataloader] processed data size : {len(self.stacked_data['observations'])}")
        self.dataset_size = len(self.stacked_data['observations'])


    def get_rand_batch(self, batch_size=None):
        return super().get_rand_batch(batch_size)

    def get_all_batch(self, batch_size=None, pool_key=None):
        if len(self.stacked_data['observations']) < 5 :
            print("[dataloader] data is less than 5")
            return None
        
        if pool_key == None :
            yield from super().get_all_batch(batch_size)
            return
    
        unique_pools = np.unique(self.stacked_data[pool_key])
        # print("sasdfs", unique_pools)
        for pool_idx in unique_pools:
            indices = np.where(self.stacked_data[pool_key] == pool_idx)[0]
            np.random.shuffle(indices)
            
            num_batches = len(indices) // batch_size
            remainder = len(indices) % batch_size

            for i in range(num_batches):
                batch_indices = indices[i * batch_size: (i + 1) * batch_size]
                batch_data = {key: self.stacked_data[key][batch_indices] for key in self.stacked_data.keys()}
                yield pool_idx, batch_data

            if remainder > 0:
                pad_indicies  = np.random.choice(indices, batch_size - remainder)
                final_indices = np.concatenate([indices[-remainder:], pad_indicies])
                final_batch_data = {key: self.stacked_data[key][final_indices] for key in self.stacked_data.keys()}
                yield pool_idx, final_batch_data

class MemoryPoolDataloaderSparse(MemoryPoolDataloader) :
    def __init__(
            self, 
            remove_skill = None,
            **kwargs,
        ) -> None:
        print("[Sparse mode MemoryPoolDataloader]")
        super().__init__(**kwargs)
        self.remove_skill=remove_skill
        print(f"[Sparse mode MemoryPoolDataloader] remove skill : {self.remove_skill}")
        print(f"[Sparse mode MemoryPoolDataloader] data size : {len(self.stacked_data['observations'])}")
        for key in self.stacked_data.keys() :
            self.stacked_data[key] = self.stacked_data[key][self.stacked_data['skills'] != self.remove_skill]
        print(f"[Sparse mode MemoryPoolDataloader] processed data size : {len(self.stacked_data['observations'])}")

class TemporalDataloader(BaseDataloader) :
    def __init__(
            self,
            data_paths=[
                'data/kitchen/kitchen_skill_appended.pkl', 
            ],  
            skill_embedding_path='data/continual_dataset/evolving_kitchen/kitchen_lang_embedding.pkl',
            semantic_flag=False,
            skill_exclude:str=None, 
            exclude_mode:str='skill', # ['episode', 'skill']
            few_shot_len=None,
            few_random_flag=False,

            # tempoarl hyperparameter
            temporal_aug_range = 4,
            temporal_ratio = 2,
        ) :
        super().__init__(  
            data_paths=data_paths,
            skill_embedding_path=skill_embedding_path,
            semantic_flag=semantic_flag,
            skill_exclude=skill_exclude,
            exclude_mode=exclude_mode,
            few_shot_len=few_shot_len,
            few_random_flag=few_random_flag,
        )

        self.temporal_ratio = temporal_ratio # original 2 ( 1: 1 ) 4 means 1:3    
        self.temporal_aug_range = temporal_aug_range

        
        # if type(skill_exclude) == str :
        #     skill_exclude = [skill_exclude]

        if few_shot_len != None :
            if few_random_flag is True :
                # self.cut_data_rand(few_shot_len)
                raise NotImplementedError
            else :
                self.cut_data(few_shot_len)

    def cut_data(self, few_shot_len) :
        few_episode_id = self.stacked_data['terminals'].nonzero()[0][few_shot_len] + 1
        self.stacked_data = {
            key : self.stacked_data[key][:few_episode_id] for key in self.stacked_data.keys()
        }
        self.episode_boundary = self.episode_boundary[:few_episode_id]
        self.dataset_size = len(self.stacked_data['observations'])
        print(f"dataset size : {self.dataset_size}")

    def process_data(self) :
        super().process_data()
        print('kitchen <Temporal> Dataloader processed!')
        
    def get_rand_batch(self, batch_size=None) :
        # TODO batch original ratio condition part
        batch_seg = batch_size // self.temporal_ratio

        # select eval episodes in episode pool
        epi_indicies = np.random.randint( 0, self.dataset_size , batch_seg )

        normal_batch = {
            key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
        }
        normal_batch['offsets'] = np.zeros(batch_seg, dtype=np.int32)

        # select offset in episode pool
        aug_offset = np.random.randint( -self.temporal_aug_range, self.temporal_aug_range+1, batch_seg)
        aug_indicies = epi_indicies + aug_offset

        # process out of boundary indicies
        aug_indicies = np.clip( 
            aug_indicies, 
            self.episode_boundary[epi_indicies,0], 
            self.episode_boundary[epi_indicies,1],
        )
    
        # augmented temporal batches 
        temporal_batch = {
            key : self.stacked_data[key][aug_indicies] for key in self.stacked_data.keys()
        }
        temporal_batch['actions'] = normal_batch['actions'].copy()
        temporal_batch['offsets'] = aug_indicies - epi_indicies

        # return batch processing 
        return_batch = {
            key : np.concatenate( [normal_batch[key], temporal_batch[key]] ) for key in normal_batch.keys()
        }
        return return_batch 

    def get_all_batch(self, batch_size=None) :
        # TODO batch original ratio condition part ONGOING
        batch_seg = batch_size // self.temporal_ratio

        # select eval episodes in episode pool
        indicies = np.arange( self.dataset_size )
        np.random.shuffle(indicies)
        for i in range(0,self.dataset_size,batch_seg) :
            epi_indicies = indicies[i:i+batch_seg]

            batches = []

            normal_batch = {
                key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
            }
            normal_batch['offsets'] = np.zeros(epi_indicies.shape, dtype=np.int32)
            batches.append(normal_batch)
            for j in range(self.temporal_ratio-1) : 
                # select offset in episode pool
                aug_offset = np.random.randint( -self.temporal_aug_range, self.temporal_aug_range+1, epi_indicies.shape[0])
                aug_indicies = epi_indicies + aug_offset

                # process out of boundary indicies
                aug_indicies = np.clip( 
                    aug_indicies, 
                    self.episode_boundary[epi_indicies,0], 
                    self.episode_boundary[epi_indicies,1],
                )
                # augmented temporal batches 
                temporal_batch = {
                    key : self.stacked_data[key][aug_indicies] for key in self.stacked_data.keys()
                }
                temporal_batch['actions'] = normal_batch['actions'].copy()
                temporal_batch['offsets'] = aug_offset
                batches.append(temporal_batch)

            # return batch processing 
            return_batch = {
                key : np.concatenate( [batch[key] for batch in batches] ) for key in normal_batch.keys()
            }
            yield return_batch

    def get_normal_all_batch(self, batch_size=None) :
        # select eval episodes in episode pool
        indicies = np.arange( self.dataset_size )
        for i in range(0,self.dataset_size,batch_size) :
            epi_indicies = indicies[i:i+batch_size]

            normal_batch = {
                key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
            }
            normal_batch['offsets'] = np.zeros(epi_indicies.shape, dtype=np.int32)

            return_batch = normal_batch
            yield return_batch

    def get_normal_rand_batch(self, batch_size=None) :
        # select eval episodes in episode pool
        indicies = np.arange( self.dataset_size )
        np.random.shuffle(indicies)
        for i in range(0,self.dataset_size,batch_size) :
            epi_indicies = indicies[i:i+batch_size]
            normal_batch = {
                key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
            }
            normal_batch['offsets'] = np.zeros(epi_indicies.shape, dtype=np.int32)

            return_batch = normal_batch
            return return_batch

    def get_kmeans_batch(self, batch_size=None) :
        batch = super().get_kmeans_batch(batch_size)
        batch['offsets'] = np.zeros(batch_size, dtype=np.int32)
        return batch

class ClusterDataloader(BaseDataloader) :
    def __init__(
            self,
            aug_ratio=4,
            cluster_size=100,
            **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.aug_ratio = aug_ratio
        self.cluster_size = cluster_size    

        self.similarity_process()

    def similarity_process(self) :
        # basic distance based similarity
        print("similarity processing start with length : ", len(self.stacked_data['actions']))
        def k_closest_rows(data, K):
            squared_norms = np.sum(data**2, axis=1).reshape(-1, 1)
            dot_products = np.dot(data, data.T)
            
            distances = squared_norms - 2 * dot_products + squared_norms.T
            
            sorted_indices = np.argsort(distances, axis=1)
            
            return sorted_indices[:, 1:K+1]
        
        sim_idx = k_closest_rows(self.stacked_data['actions'], self.cluster_size-1)
        sim_actions = self.stacked_data['actions'][sim_idx]
        sim_actions = np.concatenate([self.stacked_data['actions'][:,None,:], sim_actions], axis=1)

        self.stacked_data['actions'] = sim_actions
    
    ## batch with similar actions ##
    def get_rand_batch(self, batch_size=None) :
        batch_seg = batch_size // self.aug_ratio
        aug_seg = batch_size - batch_seg

        epi_indicies = np.random.randint( 0, self.dataset_size , batch_seg )
        aug_indicies = np.tile( epi_indicies, self.aug_ratio-1 ) 

        normal_batch = {
            key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
        }
        action_cluster = normal_batch['actions']
        normal_batch['actions'] = action_cluster[:,0,:]
        normal_batch['offsets'] = np.zeros(batch_seg, dtype=np.int32)

        # offset sampling (TODO work on ratio)
        aug_offset = np.random.randint( 0, self.cluster_size, aug_seg)

        aug_batch = {
            key : self.stacked_data[key][aug_indicies] for key in self.stacked_data.keys()
        }
        action_cluster = np.tile(action_cluster[:,:,:], (self.aug_ratio-1,1,1))
        aug_batch['actions'] = action_cluster[np.arange(aug_seg),aug_offset]
        aug_int_offset = (aug_offset/100*8).astype(np.int32) # TEMPORAL TODO
        aug_batch['offsets'] = aug_int_offset

        # return batch processing 
        return_batch = {
            key : np.concatenate( [normal_batch[key], aug_batch[key]] ) for key in normal_batch.keys()
        }
        return return_batch 

    def get_all_batch(self, batch_size=None):
        batch_seg = batch_size // self.aug_ratio

        indicies = np.arange( self.dataset_size )
        np.random.shuffle(indicies)

        for i in range(0,self.dataset_size,batch_seg) :
            epi_indicies = indicies[i:i+batch_seg]
            aug_indicies = np.tile( epi_indicies, self.aug_ratio-1 )
            aug_seg = len(aug_indicies)

            normal_batch = {
                key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
            }
            action_cluster = normal_batch['actions']
            normal_batch['actions'] = action_cluster[:,0,:]
            normal_batch['offsets'] = np.zeros(len(epi_indicies), dtype=np.int32)

            # offset sampling (TODO work on ratio)
            aug_offset = np.random.randint( 0, self.cluster_size, aug_seg)

            aug_batch = {
                key : self.stacked_data[key][aug_indicies] for key in self.stacked_data.keys()
            }
            action_cluster = np.tile(action_cluster[:,:,:], (self.aug_ratio-1,1,1))
            aug_batch['actions'] = action_cluster[np.arange(aug_seg),aug_offset]
            
            aug_int_offset = (aug_offset/100*8).astype(np.int32) # TEMPORAL TODO
            aug_batch['offsets'] = aug_int_offset

            # return batch processing 
            return_batch = {
                key : np.concatenate( [normal_batch[key], aug_batch[key]] ) for key in normal_batch.keys()
            }
            yield return_batch 

    def get_normal_rand_batch(self, batch_size=None) :
        batch_seg = batch_size 
        epi_indicies = np.random.randint( 0, self.dataset_size , batch_seg )
        normal_batch = {
            key : self.stacked_data[key][epi_indicies] for key in self.stacked_data.keys()
        }
        action_cluster = normal_batch['actions']
        normal_batch['actions'] = action_cluster[:,0,:]
        normal_batch['offsets'] = np.zeros(batch_seg, dtype=np.int32)
        return normal_batch 

# if __name__ == '__main__' :

#     data_name = 'kettle'
#     data_path = f'data/continual_dataset/evolving_kitchen/raw_skill/{data_name}.pkl'
#     with open(data_path, 'rb') as f  :
#         dataset = pickle.load(f)

#     for key in dataset.keys() :
#         print(key, len(dataset[key]))
#         if type(dataset[key][0]) == np.ndarray : 
#             print(dataset[key][0].shape)

#     dl = TemporalDataloader(
#         data_paths=[
#             data_path
#         ],
#     )
    
#     pass
