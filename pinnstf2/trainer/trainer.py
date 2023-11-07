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
        self.callback_metrics[loss_name] = loss#.numpy()
        
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
        
        self.callback_metrics[loss_name] = loss#.numpy()
        
        if extra_variables:
            for key, value in extra_variables.items():
                self.callback_metrics[key] = value.numpy()
                            
    def initalize_tqdm(self, max_epochs, mode='training', leave=True):
        """
        Initialize and return a tqdm progress bar object.

        :param max_epochs: Maximum number of epochs for which the progress bar will run.
        :param mode: Mode can be `training` or `validation`.
        :param leave: Ensures that once the bar is closed, it doesn't occupy space in the console.
        :return: Initialized tqdm progress bar object.
        """
        
        return tqdm(
                total = max_epochs,
                bar_format = mode +": {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, "
                "{rate_fmt}{postfix}, "
                "{desc}]",
                leave=leave,
                )

    def next_batch(self, dataloader, current_index, dataset_size, shuffle=True):
        """
        This function will return the next batch of data and current index.
        
        :param dataloader: The dictionary of dataloaders.
        :param current_index: The list indicating the current index.
        :param dataset_size: The list indicating the size of each dataset.
        :param shuffle: The flag for shuffling the data.

        :return: A new dictionary containing batch data and a list showing current index.
        """
        
        batch_data = {}
        for i, (key, data) in enumerate(dataloader.items()):  
            if shuffle:
                batch_data[key] = data[current_index[i]:current_index[i] + self.batch_size]
                current_index[i] = current_index[i] + self.batch_size
                
                if current_index[i] + self.batch_size >= dataset_size[i]:
                        data.shuffle()
                        current_index[i] = 0 
            else:
                if current_index[i] + self.batch_size >= dataset_size[i]:
                    batch_data[key] = data[current_index[i]:dataset_size[i]]
                else:
                    batch_data[key] = data[current_index[i]:current_index[i] + self.batch_size]
                    current_index[i] = current_index[i] + self.batch_size
                    
        return batch_data, current_index
        
    def initalize_batch_tracker(self, dataloader, shuffle=True):
        """
        Initialize tracking for batch processing.
        
        :param dataloader: The dictionary of dataloaders.
        :param shuffle: The flag for shuffling the data.

        :return: A list showing current index and a list indicating the size of datasets.
        """
        
        current_index = []
        dataset_size = []

        for i, (key, data) in enumerate(dataloader.items()):  
            current_index.append(0)
            dataset_size.append(len(data))
            if shuffle:
                data.shuffle()

        return current_index, dataset_size

    def prepare_model(self, model, datamodule):
        """
        Fix function mappings.
        
        :param datamodule: :param datamodule: The data module providing the data.
        """

        # Store validation solution names and function mappings in the model
        model.val_solution_names = datamodule.val_solution_names
        model.function_mapping = datamodule.function_mapping

        return model
    
    def fit(self, model, datamodule):    
        """
        Main function to fit the model on the provided data.

        :param model: The PINNModule.
        :param datamodule: The data module providing the data.
        """

        # Prepare the data for training and validation
        datamodule.setup('fit')
        datamodule.setup('val')

        # Load training and validation data using dataloaders
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        self.batch_size = datamodule.batch_size
        
        model = self.prepare_model(model, datamodule)

        
        # Set up the dataset for batch training
        if datamodule.batch_size is not None:
            self.train_current_index, self.train_dataset_size = self.initalize_batch_tracker(train_dataloader)
            self.eval_current_index, self.eval_dataset_size = self.initalize_batch_tracker(val_dataloader, shuffle=False)
        
        # Initialize the progress bar if enabled
        if self.enable_progress_bar:
            self.pbar = self.initalize_tqdm(self.max_epochs)
        
        for epoch in range(self.current_epoch, self.max_epochs):
            
            start_time=time.time()

            self.train_loop(model, train_dataloader)
            
            elapsed_time = time.time() - start_time
            self.time_list.append(elapsed_time)
        
            # Perform validation at specified intervals
            if epoch % self.check_val_every_n_epoch == 0:
                self.eval_loop(model, val_dataloader)

        if self.enable_progress_bar:
            self.pbar.close()

    
    def train_loop(self, model, train_dataloader):
        """
        Training loop. 

        :param model: The PINNModule.
        :param train_dataloader: The training data.
        """
        
        # Process the data in batches if batch size is specified
        if self.batch_size is not None:
            
            train_data, self.train_current_index = self.next_batch(train_dataloader,
                                                                   self.train_current_index,
                                                                   self.train_dataset_size)
            loss, extra_variables = model.train_step(train_data)
            
        else:
            # If no batching is used, pass the entire dataloader to the train_step
            loss, extra_variables = model.train_step(train_dataloader)

        self.set_callback_metrics('train/loss', loss.numpy(), extra_variables)
        
        if self.enable_progress_bar:
            self.pbar.update(1)
            description = self.callback_pbar('train/loss', loss.numpy(), extra_variables)
            self.pbar.set_description(description)
            self.pbar.refresh() 

    
    def eval_loop(self, model, eval_dataloader, mode='validation'):
        """
        Evaluation loop. 

        :param model: The PINNModule.
        :param eval_dataloader: The validation data.
        """

        
        
        if self.batch_size is not None:

            iter = (self.eval_dataset_size[0]//self.batch_size) + 1
            
            if self.enable_progress_bar:
                self.eval_pbar = self.initalize_tqdm(iter, mode=mode, leave=False)
                
            loss = []
            error_dict = {name: [] for name in model.val_solution_names}
            
            
            for i in range(iter):
                eval_data, self.eval_current_index = self.next_batch(eval_dataloader,
                                                                    self.eval_current_index,
                                                                    self.eval_dataset_size,
                                                                    shuffle=False)
                loss_i, error_dict_i = model.validation_step(eval_data)
                loss.append(loss_i.numpy())
                for key, error_i in error_dict_i.items():
                    error_i = error_i.numpy()
                    if np.isfinite(error_i):
                        error_dict[key].append(error_i)

                if self.enable_progress_bar:
                    self.eval_pbar.update(1)
                    descriptions = [self.callback_pbar('val/loss', loss[i])]
                    self.eval_pbar.set_postfix_str(descriptions)
                    self.eval_pbar.refresh() 
            self.eval_current_index = [0 for i in range(len(eval_dataloader))]
            if self.enable_progress_bar:
                self.eval_pbar.clear()
                self.eval_pbar.close()
        else:
            loss, error_dict = model.validation_step(eval_dataloader)

        loss = np.mean(loss)
        self.set_callback_metrics('val/loss', loss)
        
        if self.enable_progress_bar:
            descriptions = [self.callback_pbar('val/loss', loss)]
            for error_name in model.val_solution_names:
                error_dict[error_name] = np.mean(error_dict[error_name])
                descriptions.append(self.callback_pbar(f'val/error_{error_name}',
                                                       error_dict[error_name]))
            
            # Join all descriptions into a single string and set it
            full_description = ', '.join(descriptions)
            self.pbar.set_postfix_str(full_description)
            self.pbar.refresh() 
        return loss, error_dict

    def pred_loop(self, model, dataloader):
        """
        Prediction loop. 

        :param model: The PINNModule.
        :param dataloader: The prediction data.
        """
        preds = {}
        
        if self.batch_size is not None:
            
            iter = (self.pred_dataset_size[0]//self.batch_size) + 1
            
            if self.enable_progress_bar:
                self.pred_pbar = self.initalize_tqdm(iter, mode='prediction', leave=False)
            
            for i in range(iter):
                pred_data, self.pred_current_index = self.next_batch(dataloader,
                                                                     self.pred_current_index,
                                                                     self.pred_dataset_size,
                                                                     shuffle=False)
                preds_i = model.predict_step(pred_data)

                for key, value in preds_i.items():
                    if key in preds.keys():
                        preds[key] = np.concatenate(preds[key], preds_i[key].numpy(), 1)
                    else:
                        preds[key] = preds_i[key].numpy()
                        
                if self.enable_progress_bar:
                    self.pred_pbar.update(1)
                    self.pred_pbar.refresh() 
            
            self.pred_current_index = [0 for i in range(len(dataloader))]
            
            if self.enable_progress_bar:
                self.pred_pbar.clear()
                self.pred_pbar.close()
        else:
            preds = model.predict_step(dataloader)
            for sol_key, pred in preds.items():
                preds[sol_key] = pred.numpy()
                
        return preds

    def validate(self, model, datamodule):
        """
        Validate the model using the provided data module.

        :param model: The PINNModule.
        :param datamodule: The data module providing the validation data.
        :return: Tuple of loss and error dictionary from validation.
        """
        datamodule.setup('val')
        val_dataloader = datamodule.val_dataloader()
        model = self.prepare_model(model, datamodule)
        
        loss, error_dict = self.eval_loop(model, val_dataloader)

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
        pred_dataloader = datamodule.predict_dataloader()
        model = self.prepare_model(model, datamodule)

        if datamodule.batch_size is not None:
            self.pred_current_index, self.pred_dataset_size = self.initalize_batch_tracker(pred_dataloader, shuffle=False)

        
        preds = self.pred_loop(model, pred_dataloader)
        
        return preds

    def test(self, model, datamodule):
        """
        Test the model using the provided data module.

        :param model: The PINNModule.
        :param datamodule: The data module providing the test data.
        :return: Tuple of loss and error dictionary from testing.
        """
        
        datamodule.setup('test')
        test_dataloader = datamodule.test_dataloader()
        model = self.prepare_model(model, datamodule)

        if datamodule.batch_size is not None:
            self.eval_current_index, self.eval_dataset_size = self.initalize_batch_tracker(test_dataloader, shuffle=False)

        loss, error_dict = self.eval_loop(model, test_dataloader, mode='test')

        for key, error in error_dict.items():   
            self.set_callback_metrics(f'test/error_{error}', error)
        
        return loss, error_dict