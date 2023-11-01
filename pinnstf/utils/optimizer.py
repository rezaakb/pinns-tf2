import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm


class LossAndFlatGradient:
    """
    A helper class to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables (List[tf.Variable]): Trainable variables.
        build_loss (function): A function to build the loss function expression.
        pbar (tqdm.tqdm): An instance of tqdm progress bar for tracking progress.
        callback_pbar (function): A function to generate a progress bar description.

    Methods:
        __init__(self, trainable_variables, build_loss, pbar, callback_pbar):
            Initializes the LossAndFlatGradient helper class.
        __call__(self, weights_1d):
            A function that can be used by tfp.optimizer.lbfgs_minimize.
        evaluation(self, weights_1d):
            A function that can be used by tfp.optimizer.lbfgs_minimize.
        set_flat_weights(self, weights_1d):
            Sets the weights with a 1D tf.Tensor.
        to_flat_weights(self, weights):
            Returns a 1D tf.Tensor representing the `weights`.
    """

    def __init__(self, model, data, pbar, callback_pbar):
        self.trainable_variables = model.trainable_variables
        self.model = model
        self.data = data
         
        self.pbar = pbar
        self.callback_pbar = callback_pbar

        # Shapes of all trainable parameters
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.indices_index = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.indices_index.append([count, count+n])
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        Args:
           weights_1d: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        loss, grads, extra = self.evaluation(weights_1d)

        self.pbar.update(1/4)
        self.pbar.set_description_str(self.callback_pbar(loss, extra))

        return loss, grads

    @tf.function#(jit_compile=True)
    def evaluation(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        Args:
           weights_1d: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """

        # Set the weights
        self.set_flat_weights(weights_1d)
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss, extra = self.model.train_step(self.data)
        # Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.dynamic_stitch(self.indices, grads)

        return loss, grads, extra

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.

        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        '''
        for i in range(self.n_tensors):
            self.trainable_variables[i].assign(tf.reshape(weights_1d[self.indices_index[i][0]:self.indices_index[i][1]], self.shapes[i]))
        '''
        weights = tf.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        for i, (shape) in enumerate(self.shapes):
            self.trainable_variables[i].assign(tf.reshape(weights[i], shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.

        Args:
            weights: A list of tf.Tensor representing the weights.

        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)


def lbfgs_minimize(
    model,
    data,
    callback_pbar,
    cfg_opt=None,
    previous_optimizer_results=None,
):
    """
    TensorFlow interface for tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables (List[tf.Variable]): Trainable variables, also used as the initial position.
        build_loss (function): A function to build the loss function expression.
        callback_pbar (function): A function to generate a progress bar description.
        cfg_opt (object): Configuration options for the optimizer.
        previous_optimizer_results (object): Results from previous optimizer runs.

    Returns:
        object: The results of the optimization process.

    Raises:
        None.
    """

    pbar = tqdm(
        total=5000,
        bar_format="{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, "
        "{rate_fmt}{postfix}, "
        "{desc}]",
    )

    func = LossAndFlatGradient(model, data, pbar, callback_pbar)
    initial_position = None
    if previous_optimizer_results is None:
        initial_position = func.to_flat_weights(model.trainable_variables)

    results = tfp.optimizer.lbfgs_minimize(
        func,
        initial_position=initial_position,
        previous_optimizer_results=previous_optimizer_results,
        num_correction_pairs=50,
        tolerance=1e-5,
        x_tolerance=0,
        f_relative_tolerance=2.220446049250313e-16,
        max_iterations=5000,
        parallel_iterations=1,
        max_line_search_iterations=50,
    )

    pbar.close()

    func.set_flat_weights(results.position)
    return results