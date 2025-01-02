from clus.trainer.base_trainer import *
from clus.models.peftpool.dual_l2m import LoRAPoolModel, DualLoRAPoolModel
from tqdm import tqdm
from clus.config.algo_config import ExpAlgorithmConfig

class MemoryPoolCLTrainer(ContinualTrainer) :
    def __init__(
            self, 
            algorithm = ExpAlgorithmConfig(),
            **kwargs,    
        ) -> None:
        super().__init__(**kwargs)
        self.peft_pool_model_cls = algorithm.peft_pool_model_cls
        self.peft_pool_model_kwargs = algorithm.peft_pool_model_kwargs

    def init_phase_model(self, phase: int):
        if phase == 0 :
            super().init_phase_model(phase)
            self.peft_pool_model_kwargs['model'] = self.model
            print(f'[phase {phase}] LoRA Pool Model Wrapped')
            self.model = self.peft_pool_model_cls(
                **self.peft_pool_model_kwargs,
            ) 
            self.initial_params = self.model.model.train_state.params
        else :
            if self.model.tail_flag == True :
                print(f'[phase {phase}] tail_flag is True')
                if phase < 10 : 
                    self.model.model.train_state = self.model.model.train_state.replace(params=self.initial_params)
                else : 
                    with open(f'{self.model_base_path}/model_{phase-10}.pkl', 'rb') as f :
                        self.model_loaded = cloudpickle.load(f)
                        loaded_params = self.model_loaded.model.train_state.params
                        self.model.model.train_state = self.model.model.train_state.replace(params=loaded_params)
            
            print(f'[phase {phase}] LoRA Pool Model Reinitialized')
            self.model.reinit_optimizer()
        # memory leackage problem on loading the model again
        self.model.next_phase_processing(phase)

    def train_phase_model(self, phase: int):
        print(f'[phase {phase}] key_counts : ', self.model.key_module.key_appear_count)

        batch_size = self.exp_config['batch_size']
        for epoch in tqdm(range(self.exp_config['phase_epoch'])) :
            if epoch % self.exp_config['eval_epoch'] == 0 and epoch != 0 :
                self.eval_phase_model(phase=phase, epoch=epoch)

            total_loss = self.model.train_model(
                self.dataloader, 
                batch_size=batch_size,
                first_epoch=(epoch == 0),
            )
            wandb.log({'loss' : total_loss.item()})
            if epoch % self.exp_config['eval_epoch'] == 0 :
                print(f'[epoch {epoch}]train/loss : ', total_loss)
        last_epoch = self.exp_config['phase_epoch']
        print(f'[epoch {last_epoch}]train/loss : ', total_loss)
        # self.model.first_phase = False # setting for key reverse probability
        self.eval_phase_model(phase=phase, epoch=epoch)

    def eval_phase_model(
            self,
            phase: int,
            epoch: int,
        ) :
        print(f'[phase {phase}] evaluation')
        return self.continual_scenario.evaluation_function(self.model, log_stage=phase)

    