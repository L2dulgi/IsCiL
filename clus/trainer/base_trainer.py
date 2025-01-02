import wandb
import cloudpickle

import cloudpickle
from tqdm import tqdm 
from clus.models.utils.train_state import *
from clus.models.model.basic import *
from clus.models.model.cdm import ConditionalDiffusion

from clus.utils.utils import create_directory_if_not_exists, create_directory
from clus.env.cl_scenario import ContinualScenario , MMEvaluator

class ContinualReplayBuffer() :
    def __init__(
            self,
            buffer_size:int=1000,
            sample_data=None,
        ) -> None:
        
        self.buffer_size = buffer_size
        self.sample_data = sample_data  
        self.buffer_data = {
            k : [] for k in self.sample_data.keys()    
        }
        self.phase_buffer = []

    def add(self, data, phase_idx:int=0) :
        for k in self.buffer_data.keys() :
            self.buffer_data[k].extend(data[k])
        
        for k in self.buffer_data.keys() :
            while len(self.buffer_data[k]) > self.buffer_size :
                self.buffer_data[k].pop(0)
        
        self.stacked_data = {
            k : np.stack(self.buffer_data[k]) for k in self.buffer_data.keys()
        }

        k = list(self.stacked_data.keys())[0] 
        self.data_size = len(self.stacked_data[k])

    def update_stack(self) :
        self.stacked_data = {
            k : np.stack(self.buffer_data[k]) for k in self.buffer_data.keys()
        }
        k = list(self.stacked_data.keys())[0] 
        self.data_size = len(self.stacked_data[k])

    def get_rand_batch(self, batch_size:int) :
        indicies = np.random.choice( self.data_size , batch_size )
        batch = {
            key : self.stacked_data[key][indicies] for key in self.stacked_data.keys()
        }
        return batch

class BaseTrainer() :
    def __init__(self) -> None:
        pass

    # <<<<< core function >>>>> #
    def continual_train(
            self,
        ) :
        if self.logging_config['logger'] == 'wandb' :
            wandb.init(
                project=self.logging_config['proj_name'],
                name=self.logging_config['exp_name'],
            )

        for phase in range(self.continual_scenario.phase_num) :
            print(f'[phase {phase}] start' + 'V'*20)
            self.phase_train(phase)
            print(f'[phase {phase}] ended' + '^'*20)

        if self.logging_config['logger'] == 'wandb' :
            wandb.finish()

    # <<<<< phase function >>>>> #
    def phase_train(
            self,
            phase:int,
        ) :
        self.load_phase_data(phase)
        self.init_phase_model(phase)
        self.train_phase_model(phase)
        self.save_phase_model(phase)
        self.process_phase_data(phase)
    
    def load_phase_data(
            self,
            phase:int,
        ) :
        raise NotImplementedError
    
    def init_phase_model(
            self,
            phase:int,
        ) :
        raise NotImplementedError
    
    def train_phase_model(
            self,
            phase:int,
        ) :
        raise NotImplementedError
    
    def save_phase_model(
            self,
            phase:int,
        ) :
        raise NotImplementedError
    
    def process_phase_data(
            self,
            phase:int,
        ) :
        raise NotImplementedError
    
DEFAULT_MODEL =  {
    'model_cls' : MLPModule,
    'model_kwargs' : {
        'input_config' : None,
        'model_config' : {
            'hidden_size' : 512,
            'out_shape' : None,
            'num_hidden_layers' : 8,
            'dropout' : 0.0,
        },
    },
}

DEFAULT_EXP = {
    'phase_epoch' : 250,
    'eval_epoch' : 50,
    'batch_size' : 1024,
    'eval_env' : True,
    'base_path' : './data/exptest', # base path for saving items
    'phase_optim' : 're_initialize',
    'replay_method' : 'kmeans',  # 'kmeans' or 'random'
    'phase_batch_sz' : 100, 
    'init_model_path' : None, 
}

class ContinualTrainer(BaseTrainer) :
    def __init__(
            self,
            continual_scenario:ContinualScenario=None,
            model_config=None,
            exp_config=None,
            logging_config=None,
            adapt_from_zero:bool=False,
        ) -> None:
        super().__init__()  

        ## initialize configures ## 
        self.continual_scenario = continual_scenario
        self.adapt_from_zero = adapt_from_zero
        if self.continual_scenario is None :
            self.continual_scenario = ContinualScenario()

        self.model_config = model_config
        if self.model_config is None :
            self.model_config = DEFAULT_MODEL

        self.exp_config = exp_config
        if self.exp_config is None :
            self.exp_config = DEFAULT_EXP

        self.logging_config = logging_config 
        if self.logging_config is None :
            self.logging_config = {
                'logger' : 'wandb',
                'proj_name' : 'Conttest',
                'exp_name' : 'testa',
                'logging' : True,
            } 

        print("="*20 + "configures" + "="*20)        
        print(f'[model config] {self.model_config}')
        print(f'[exp config] {self.exp_config}')
        print(f'[logging config] {self.logging_config}')
        print("="*50)

        dfc = self.continual_scenario.data_feature_configs
        self.out_config = dfc['actions']
        self.input_config = { # TODO make input config by model configure
            'x' : (1, dfc['observations']),
        }
        if self.model_config['model_cls'] == ConditionalDiffusion  :
            self.input_config = {
                'context' : (1,1, dfc['observations']),
                'hidden_states' : (1, 1, dfc['actions']),
            }
        else : # not TCD
            if 'out_shape' not in self.model_config['model_kwargs']['model_config'].keys() \
                 or self.model_config['model_kwargs']['model_config']['out_shape'] is None:
                self.model_config['model_kwargs']['model_config']['out_shape'] = self.out_config

        self.model_config['model_kwargs']['input_config'] = self.input_config
        # Temporal <<<<<<<<<<<<<<<<

        self.model_base_path = f'{self.exp_config["base_path"]}/models'
        create_directory(self.model_base_path)

        if self.exp_config['eval_env'] is False : 
            self.continual_scenario.evaluation_function = lambda x : 0
            
        self.model = None
        self.replay_buffer = None


    # <<<<< core function >>>>> #
    def continual_train(
            self,
        ) :
        if self.logging_config['logger'] == 'wandb' :
            wandb.init(
                project='iscil',
                name=self.logging_config['exp_name'],
            )

        for phase in range(self.continual_scenario.phase_num) :
            print(f'[phase {phase}] start' + 'V'*20)
            self.phase_train(phase)
            print(f'[phase {phase}] ended' + '^'*20)

        if self.logging_config['logger'] == 'wandb' :
            wandb.finish()

    # <<<<< phase function >>>>> #
    def phase_train(
            self,
            phase:int,
        ) :
        self.load_phase_data(phase)
        self.init_phase_model(phase)
        self.train_phase_model(phase)
        self.save_phase_model(phase)
        self.process_phase_data(phase)
    
    def load_phase_data(
            self,
            phase:int,
        ) :
        self.dataloader = self.continual_scenario.get_phase_data(phase)
        print(f'[phase {phase}] data loaded')
        print(self.continual_scenario.phase_configures[phase] )
        
    def init_phase_model(
            self,
            phase:int,
        ) :
        self.full_load = False
        if self.model is None or self.adapt_from_zero == True:
            if 'init_model_path' in self.exp_config.keys() and self.exp_config['init_model_path'] is not None :
                print(f'[phase {phase}] model loaded from {self.exp_config["init_model_path"]}')
                with open(self.exp_config['init_model_path'], 'rb') as f :
                    self.model = cloudpickle.load(f)


            else :
                self.model = self.model_config['model_cls'](
                    **self.model_config['model_kwargs'],
                )
        else :
            with open(f'{self.model_base_path}/model_{phase-1}.pkl', 'rb') as f :
                self.model_loaded = cloudpickle.load(f)
                if self.full_load == True :
                    self.model = self.model_loaded
                else :
                    loaded_params = self.model_loaded.train_state.params
                    self.model.train_state = self.model_loaded.train_state.replace(params=loaded_params)

        if self.exp_config['phase_optim'] == 're_initialize'  :
            print(f'[phase {phase}] model optim re-initialized')
            self.model.reinit_optimizer()
            
        print(f'[phase {phase}] model initialized')
    
    def train_phase_model(
            self,
            phase:int,
        ) :
        batch_size = self.exp_config['batch_size']
        if self.exp_config['phase_epoch'] == 0 :
            self.eval_phase_model(phase=phase, epoch=0)
            return 0


        for epoch in tqdm(range(self.exp_config['phase_epoch'])) :
            total_loss = 0
            for b_count, batch in enumerate(self.dataloader.get_all_batch(batch_size)) :
                if self.replay_buffer != None :
                    replay_batch_size = len(batch['observations'])
                    replay_batch = self.replay_buffer.get_rand_batch(replay_batch_size)
                    for k in batch.keys() :
                        batch[k] = np.concatenate([batch[k], replay_batch[k]], axis=0)

                if isinstance(self.model, ConditionalDiffusion) :
                    cond = batch['observations']
                    x = batch['actions']
                    metric = self.model.train_model(x=x,cond=cond)
                    loss = metric[1]['train/loss']
                else : 
                    input_batch = {
                        'inputs' : batch['observations'],
                        'labels' : batch['actions'],
                    }
                    loss = self.model.train_model(input_batch)
                total_loss += loss
            total_loss /= b_count

            wandb.log({'loss' : total_loss.item()})

            if epoch % self.exp_config['eval_epoch'] == 0 and epoch != 0:
                print('train/loss : ', total_loss)
                self.eval_phase_model(phase=phase, epoch=epoch)
                
        print('train/loss : ', total_loss)
        self.eval_phase_model(phase=phase, epoch=epoch)

    def eval_phase_model(
            self,
            phase:int,
            epoch:int,
        ) :
        eval_dict = self.continual_scenario.evaluation_function(self.model)
        # save model 
        with open(f'{self.model_base_path}/model_{phase}_{epoch}.pkl', 'wb') as f :
            cloudpickle.dump(self.model, f)

    def save_phase_model(
            self,
            phase:int,
        ) :
        with open(f'{self.model_base_path}/model_{phase}.pkl', 'wb') as f :
            cloudpickle.dump(self.model, f)

    def process_phase_data(
            self,
            phase:int,
        ) :
        '''pre processing data for next phase'''
        # if replay mehtod is sequential then no usefor replay buffer
        if self.exp_config['replay_method'] == 'sequential' :
            print(f"[phase {phase}] replay method is sequential")
            return
        
        # basic experience replay
        if 'phase_batch_sz' not in self.exp_config.keys() :
            print("[process_phase_data] no replay")
            return
        else :
            if self.exp_config['phase_batch_sz'] is None or self.exp_config['phase_batch_sz'] == 0 :
                print(f"[process_phase_data] no replay")
                return
            print(f"[process_phase_data] replay with {self.exp_config['phase_batch_sz']}")
            phase_batch_sz = self.exp_config['phase_batch_sz']

        if self.replay_buffer is None :
            self.replay_buffer = ContinualReplayBuffer(
                buffer_size=1_000_000,
                sample_data=self.dataloader.get_rand_batch(16),
            )

        # if replay for full means get all data from the dataloader
        if phase_batch_sz == -1 :
            print(f"[phase {phase}] replay full data")
            replay_batch = self.dataloader.get_rand_batch(-1)
        else :
            print(f"[phase {phase}] replay random data")
            replay_batch = self.dataloader.get_rand_batch(phase_batch_sz)
        
        self.replay_buffer.add(
            replay_batch,
            phase_idx=phase,
        )

class CLUTrainer(ContinualTrainer):
    '''
    Continual Learning Unlearning Trainer.
    - Scenario is given
    '''
    def __init__(
            self,
            unlearning_algo='GA',
            **kwargs,
        ) :
        self.unlearning_algo = unlearning_algo
        self.unlearning_phases = [3,8,14,19] # 3 7 + 1 12 +2 16 + 3
        self.unlearning_count = 0
        self.unlearning_epoch = 50
        print(f'[CLUTrainer] unlearning mode : {self.unlearning_algo}')
        super().__init__(**kwargs)
    
    def train_phase_model(
            self,
            phase:int,
        ) :
        current_mode = 'learn'
        if 'mode' in self.continual_scenario.phase_configures[phase].keys() :
            current_mode = self.continual_scenario.phase_configures[phase]['mode']
            print(f'[phase {phase}] mode : {current_mode}')  

        if current_mode == 'unlearn' :
            unlearn_phase = self.unlearning_phases[self.unlearning_count]
            print(f'[phase {phase}] unlearn phase : {unlearn_phase}')
            if self.replay_buffer is not None :
                print('prev buffer size : ', self.replay_buffer.data_size)
                # print(self.replay_buffer.buffer_data['phase'])
                unlearn_phase = np.array(unlearn_phase, dtype=np.int32)

                valid_indices = np.where(self.replay_buffer.buffer_data['phase'] != unlearn_phase)[0]

                self.replay_buffer.buffer_data = {
                    k: np.array(v)[valid_indices].tolist() for k, v in self.replay_buffer.buffer_data.items()
                }
                
                self.replay_buffer.update_stack()
                print('after buffer size : ', self.replay_buffer.data_size)
                self.unlearning_count += 1

        batch_size = self.exp_config['batch_size']
        if self.exp_config['phase_epoch'] == 0 :
            self.eval_phase_model(phase=phase, epoch=0)
            return 0

        phase_epoch = self.exp_config['phase_epoch'] if current_mode == 'learn' \
                    else self.unlearning_epoch # unlearning epoch
        if self.unlearning_algo == 'ER' :
            phase_epoch = self.exp_config['phase_epoch']
            
        for epoch in tqdm(range(phase_epoch)) :
            total_loss = 0
            for b_count, batch in enumerate(self.dataloader.get_all_batch(batch_size)) :
                if self.replay_buffer != None :
                    replay_batch_size = len(batch['observations'])
                    replay_batch = self.replay_buffer.get_rand_batch(replay_batch_size)
                    if current_mode == 'learn' :
                        for k in batch.keys() : # concatenate replay batch
                            batch[k] = np.concatenate([batch[k], replay_batch[k]], axis=0)  
                if current_mode == 'learn' :
                    if isinstance(self.model, ConditionalDiffusion) :
                        cond = batch['observations']
                        x = batch['actions']
                        # no reg for learning
                        metric = self.model.train_model(x=x,cond=cond,reg=False)
                        loss = metric[1]['train/loss']
                    else : 
                        input_batch = {
                            'inputs' : batch['observations'],
                            'labels' : batch['actions'],
                        }
                        loss = self.model.train_model(input_batch)
                    total_loss += loss
                elif current_mode == 'unlearn' :
                    reg=True
                    if self.unlearning_algo in ['GA', 'ERGA', 'ER'] :
                        reg = False
                    if isinstance(self.model, ConditionalDiffusion) :
                        u_cond = batch['observations']
                        u_x = batch['actions']
                        x = replay_batch['actions'] if self.replay_buffer is not None else None
                        cond = replay_batch['observations'] if self.replay_buffer is not None else None
                        if self.unlearning_algo == 'ER' :
                            # train by only replay data
                            metric = self.model.train_model(
                                x=x, cond=cond,    
                                reg=reg,
                            )
                            loss = metric[1]['train/loss']
                        else :
                            metric = self.model.train_model(
                                x=x, cond=cond,
                                u_x=u_x,u_cond=u_cond,    
                                reg=reg,
                            )
                            loss = metric[1]['train/loss']
                    else : 
                        raise NotImplementedError
                    total_loss += loss
            total_loss /= b_count
            wandb.log({'loss' : total_loss.item()})

            if epoch % self.exp_config['eval_epoch'] == 0 and epoch != 0:
                print('train/loss : ', total_loss)
                self.eval_phase_model(phase=phase, epoch=epoch)
                
        print('train/loss : ', total_loss)
        self.eval_phase_model(phase=phase, epoch=epoch)

    def process_phase_data(
            self,
            phase:int,
        ) :
        '''pre processing data for next phase'''
        # if replay mehtod is sequential then no usefor replay buffer
        if self.exp_config['replay_method'] == 'sequential' :
            print(f"[phase {phase}] replay method is sequential")
            return
        current_mode = 'learn'
        if 'mode' in self.continual_scenario.phase_configures[phase].keys() :
            current_mode = self.continual_scenario.phase_configures[phase]['mode']
            print(f'[phase {phase}] mode : {current_mode}  no buffer')
            if current_mode == 'unlearn' :
                return

                
        # basic experience replay
        if 'phase_batch_sz' not in self.exp_config.keys() :
            print("[process_phase_data] no replay")
            return
        else :
            if self.exp_config['phase_batch_sz'] is None or self.exp_config['phase_batch_sz'] == 0 :
                print(f"[process_phase_data] no replay")
                return
            print(f"[process_phase_data] replay with {self.exp_config['phase_batch_sz']}")
            phase_batch_sz = self.exp_config['phase_batch_sz']

        if self.replay_buffer is None :
            sample_data = self.dataloader.get_rand_batch(16)
            sample_data['phase'] = np.ones(16, dtype=np.int32) * phase
            self.replay_buffer = ContinualReplayBuffer(
                buffer_size=1_000_000,
                sample_data=sample_data,
            )

        # if replay for full means get all data from the dataloader
        if phase_batch_sz == -1 :
            print(f"[phase {phase}] replay full data")
            replay_batch = self.dataloader.get_rand_batch(-1)
        else :
            print(f"[phase {phase}] replay random data")
            replay_batch = self.dataloader.get_rand_batch(phase_batch_sz)
        
        replay_batch['phase'] = np.ones(phase_batch_sz, dtype=np.int32) * phase
        self.replay_buffer.add(
            replay_batch,
            phase_idx=phase,
        )