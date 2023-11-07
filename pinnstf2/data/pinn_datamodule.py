
class PINNDataModule:
    def __init__(self,
                 train_datasets,
                 val_dataset,
                 test_dataset=None,
                 pred_dataset=None,
                 batch_size=None,
                ):
        """Initialize a `MNISTDataModule`.

        :param train_datasets: train datasets.
        :param val_dataset: validation dataset.
        """
        
        self.train_datasets = train_datasets
        self.val_datasets = val_dataset
        self.test_datasets = test_dataset
        self.pred_datasets = pred_dataset

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pred_data = None
        self.batch_size = batch_size

        self.function_mapping = {}

    
    def setup(self, stage: str):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if stage=='fit':
            if not self.train_data:
                self.train_data = {}
                self.set_mode_for_discrete_mesh()
    
                for train_dataset in self.train_datasets:
                    if self.batch_size is not None:
                        self.train_data[str(train_dataset.loss_fn)] = train_dataset
                    else:
                        self.train_data[str(train_dataset.loss_fn)] = train_dataset[:]
                    self.function_mapping[str(train_dataset.loss_fn)] = train_dataset.loss_fn
        
        if stage=='val':
            if not self.val_data:
                self.val_solution_names = []
                self.val_data = {}
                if not isinstance(self.val_datasets, list):
                    self.val_datasets = [self.val_datasets]
                for val_dataset in self.val_datasets:
                    if self.batch_size is not None:
                        self.val_data[str(val_dataset.loss_fn)] = val_dataset
                    else:
                        self.val_data[str(val_dataset.loss_fn)] = val_dataset[:]
                    self.function_mapping[str(val_dataset.loss_fn)] = val_dataset.loss_fn
                    self.val_solution_names.extend(val_dataset.solution_names)

        if stage=='test':
            if not self.test_data:
                self.test_data = {}
                if not isinstance(self.test_datasets, list):
                    self.test_datasets = [self.test_datasets]
                for test_dataset in self.test_datasets:
                    self.test_data[str(test_dataset.loss_fn)] = test_dataset[:]
                    self.function_mapping[str(test_dataset.loss_fn)] = test_dataset.loss_fn

        if stage=='pred':
            if not self.pred_data:
                self.pred_data = {}
                if not isinstance(self.pred_datasets, list):
                    self.pred_datasets = [self.pred_datasets]
                for pred_dataset in self.pred_datasets:
                    self.pred_data[str(pred_dataset.loss_fn)] = pred_dataset[:]
                    self.function_mapping[str(pred_dataset.loss_fn)] = pred_dataset.loss_fn
        
     

    def set_mode_for_discrete_mesh(self):
        """This function will figuere out which training datasets are for discrete.

        Then set the mode value that will be used for Runge–Kutta methods
        """

        mesh_idx = [
            (train_dataset.idx_t, train_dataset)
            for train_dataset in self.train_datasets
            if type(train_dataset).__name__ == "DiscreteMeshSampler"
        ]

        mesh_idx = sorted(mesh_idx, key=lambda x: x[0])

        if len(mesh_idx) == 1:
            mesh_idx[0][1].mode = "forward_discrete"
        elif len(mesh_idx) == 2:
            mesh_idx[0][1].mode = "inverse_discrete_1"
            mesh_idx[1][1].mode = "inverse_discrete_2"

        mesh_idx.clear()


        
    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.train_data

    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The train dataloader.
        """
        return self.val_data
        
    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The train dataloader.
        """
        return self.test_data

    def predict_dataloader(self):
        """Create and return the predict dataloader.

        :return: The train dataloader.
        """
        return self.pred_data 


if __name__ == "__main__":
    _ = PINNDataModule(None, None)

        