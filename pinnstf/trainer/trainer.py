import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pinnstf import utils

log = utils.get_pylogger(__name__)


class Trainer:
    
    def __init__(self,
                 max_epochs,
                 min_epochs: int=1,
                 enable_progress_bar: bool=True,
                 check_val_every_n_epoch: int = 1,
                 accelerator: str = 'cpu',
                 default_root_dir: str = "",
                 lbfgs=None,
                 devices=None):
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.enable_progress_bar = enable_progress_bar
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.callback_metrics = {}
        self.current_epoch = 0
        self.lbfgs = lbfgs
        self.accelerator = accelerator
        self.default_root_dir = default_root_dir
        self.time_list = []

    def callback_pbar(self, loss_name, loss, extra_variables=None):
        res = f"{loss_name}: {loss:.4f}"
        self.callback_metrics[loss_name] = loss.numpy()
        
        if extra_variables:
            disc = []
            for key, value in extra_variables.items():
                disc.append(f"{key}: {value:.4f}")
                self.callback_metrics[key] = value.numpy()
                
            extra_info = ', '.join(disc)
            res = f"{res}, {extra_info}"
        
        return res

    def set_callback_metrics(self, loss_name, loss, extra_variables=None):
        self.callback_metrics[loss_name] = loss.numpy()
        
        if extra_variables:
            for key, value in extra_variables.items():
                self.callback_metrics[key] = value.numpy()
                            
    def initalize_tqdm(self, max_epochs):
        return tqdm(
                total= max_epochs,
                bar_format="{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, "
                "{rate_fmt}{postfix}, "
                "{desc}]",
                )
        
    def fit(self, model, datamodule):    
        datamodule.setup('fit')
        datamodule.setup('val')
        
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        
        model.val_solution_names = datamodule.val_solution_names
        model.function_mapping = datamodule.function_mapping
        
        if self.enable_progress_bar:
            self.pbar = self.initalize_tqdm(self.max_epochs)
        
        log.info("Training with Adam Optimizer")

        self.current_index = []
        self.dataset_size = []

        if datamodule.batch_size is not None:
            for i, (key, data) in enumerate(train_dataloader.items()):  
                self.current_index.append(0)
                self.dataset_size.append(len(data))
                data.shuffle()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            
            start_time=time.time()

            if datamodule.batch_size is not None:
                
                train_data = {}
                for i, (key, data) in enumerate(train_dataloader.items()):  
                    train_data[key] = data[self.current_index[i]:self.current_index[i] + datamodule.batch_size]
                    self.current_index[i] = self.current_index[i] + datamodule.batch_size
                    
                    if self.current_index[i] + datamodule.batch_size >= self.dataset_size[i]:
                        data.shuffle()
                        self.current_index[i] = 0 
                
                loss, extra_variables = model.train_step(train_data)
                
            else:
                
                loss, extra_variables = model.train_step(train_dataloader)

            elapsed_time = time.time() - start_time
            self.time_list.append(elapsed_time)

            self.set_callback_metrics('train/loss', loss, extra_variables)
            
            if self.enable_progress_bar:
                self.pbar.update(1)
                description = self.callback_pbar('train/loss', loss, extra_variables)
                self.pbar.set_description(description)
                self.pbar.refresh() 

        

            if epoch % self.check_val_every_n_epoch == 0:
                loss, error_dict = model.validation_step(val_dataloader)
                self.set_callback_metrics('val/loss', loss)
                
                if self.enable_progress_bar:
                    descriptions = [self.callback_pbar('val/loss', loss)]
                    for error_name in model.val_solution_names:
                        descriptions.append(self.callback_pbar(f'val/error_{error_name}', error_dict[error_name]))
                    
                    # Join all descriptions into a single string and set it
                    full_description = ', '.join(descriptions)
                    self.pbar.set_postfix_str(full_description)
                    self.pbar.refresh() 
                    

            

        if self.enable_progress_bar:
            self.pbar.close()
        
        if self.lbfgs:
            log.info("Training with L-BFGS-B Optimizer")
            lbfgs_minimize(model, data, self.callback_pbar)
        

    def validate(self, model, datamodule):
        datamodule.setup('val')
        data = datamodule.val_dataloader()

        loss, error_dict = model.validation_step(data)

        for key, error in error_dict.items():   
            self.set_callback_metrics(f'val/error_{key}', error)
        
        return loss, error_dict

    def predict(self, model, datamodule):
        datamodule.setup('pred')
        data = datamodule.predict_dataloader()

        preds = model.predict_step(data)
        
        return preds

    def test(self, model, datamodule):
        datamodule.setup('test')
        data = datamodule.test_dataloader()

        log.info("Test started")

        loss, error_dict = model.test_step(data)

        for key, error in error_dict.items():   
            self.set_callback_metrics(f'test/error_{error}', error)
        
        return loss, error_dict