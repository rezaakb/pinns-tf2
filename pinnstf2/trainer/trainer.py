import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pinnstf2 import utils

log = utils.get_pylogger(__name__)


class Trainer:
    """
    Trainer Class
    """
    
    def __init__(self,
                 max_epochs,
                 min_epochs: int=1,
                 enable_progress_bar: bool=True,
                 check_val_every_n_epoch: int = 1,
                 default_root_dir: str = ""):
        """
        Initialize the Trainer class with specified training parameters.

        :param max_epochs: Maximum number of training epochs.
        :param min_epochs: Minimum number of training epochs.
        :param enable_progress_bar: Flag to enable/disable the progress bar.
        :param check_val_every_n_epoch: Frequency of validation checks within epochs.
        :param default_root_dir: Default directory for saving model-related files.
        """
        
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.enable_progress_bar = enable_progress_bar
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.callback_metrics = {}
        self.current_epoch = 0
        self.default_root_dir = default_root_dir
        self.time_list = []

    def callback_pbar(self, loss_name, loss, extra_variables=None):
        """
        Update and format the string for the tqdm progress bar.

        :param loss_name: Name of the loss function.
        :param loss: Loss value.
        :param extra_variables: Additional trainable variables to include in progress bar.
        :return: Formatted string with loss and extra variables values.
        """
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
        """
        Set callback metrics such as loss and additional variables.

        :param loss_name: Name of the loss function.
        :param loss: Loss value.
        :param extra_variables: Additional trainable variables to be logged.
        """
        
        self.callback_metrics[loss_name] = loss.numpy()
        
        if extra_variables:
            for key, value in extra_variables.items():
                self.callback_metrics[key] = value.numpy()
                            
    def initalize_tqdm(self, max_epochs):
        """
        Initialize and return a tqdm progress bar object.

        :param max_epochs: Maximum number of epochs for which the progress bar will run.
        :return: Initialized tqdm progress bar object.
        """
        
        return tqdm(
                total= max_epochs,
                bar_format="{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, "
                "{rate_fmt}{postfix}, "
                "{desc}]",
                )
        
    def fit(self, model, datamodule):    
        """
        Main function to fit the model on the provided data.

        :param model: The PINNModule.
        :param datamodule: The data module providing the training and validation data.
        """

        # Prepare the data for training and validation
        datamodule.setup('fit')
        datamodule.setup('val')

        # Load training and validation data using dataloaders
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        # Store validation solution names and function mappings in the model
        model.val_solution_names = datamodule.val_solution_names
        model.function_mapping = datamodule.function_mapping

        # Initialize the progress bar if enabled
        if self.enable_progress_bar:
            self.pbar = self.initalize_tqdm(self.max_epochs)
        
        log.info("Training with Adam Optimizer")

        # Initialize tracking for batch processing
        self.current_index = []
        self.dataset_size = []

        # Set up the dataset for batch training
        if datamodule.batch_size is not None:
            for i, (key, data) in enumerate(train_dataloader.items()):  
                self.current_index.append(0)
                self.dataset_size.append(len(data))
                data.shuffle()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            
            start_time=time.time()

            # Process the data in batches if batch size is specified
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
                # If no batching is used, pass the entire dataloader to the train_step
                loss, extra_variables = model.train_step(train_dataloader)

            elapsed_time = time.time() - start_time
            self.time_list.append(elapsed_time)

            self.set_callback_metrics('train/loss', loss, extra_variables)
            
            if self.enable_progress_bar:
                self.pbar.update(1)
                description = self.callback_pbar('train/loss', loss, extra_variables)
                self.pbar.set_description(description)
                self.pbar.refresh() 

        
            # Perform validation at specified intervals
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
        

    def validate(self, model, datamodule):
        """
        Validate the model using the provided data module.

        :param model: The PINNModule.
        :param datamodule: The data module providing the validation data.
        :return: Tuple of loss and error dictionary from validation.
        """
        datamodule.setup('val')
        data = datamodule.val_dataloader()

        loss, error_dict = model.validation_step(data)

        for key, error in error_dict.items():   
            self.set_callback_metrics(f'val/error_{key}', error)
        
        return loss, error_dict

    def predict(self, model, datamodule):
        """
        Generate predictions using the model and data module.

        :param model: The PINNModule.
        :param datamodule: The data module providing prediction data.
        :return: Predictions made by the model.
        """
        
        datamodule.setup('pred')
        data = datamodule.predict_dataloader()

        preds = model.predict_step(data)
        
        return preds

    def test(self, model, datamodule):
        """
        Test the model using the provided data module.

        :param model: The PINNModule.
        :param datamodule: The data module providing the test data.
        :return: Tuple of loss and error dictionary from testing.
        """
        
        datamodule.setup('test')
        data = datamodule.test_dataloader()

        log.info("Test started")

        loss, error_dict = model.test_step(data)

        for key, error in error_dict.items():   
            self.set_callback_metrics(f'test/error_{error}', error)
        
        return loss, error_dict