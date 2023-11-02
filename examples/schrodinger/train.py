from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import tensorflow as tf
from omegaconf import DictConfig

import pinnstf2


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstf2.utils.load_data(root_path, "NLS.mat")
    exact = data["uu"]
    exact_u = np.real(exact)
    exact_v = np.imag(exact)
    exact_h = np.sqrt(exact_u**2 + exact_v**2)
    return {"u": exact_u, "v": exact_v, "h": exact_h}


def output_fn(outputs: Dict[str, tf.Tensor],
              x: tf.Tensor,
              t: tf.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    outputs["h"] = tf.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

    return outputs


def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           t: tf.Tensor):   
    """Define the partial differential equations (PDEs)."""
    u_x, u_t = pinnstf2.utils.gradient(outputs["u"], [x, t])
    v_x, v_t = pinnstf2.utils.gradient(outputs["v"], [x, t])

    u_xx = pinnstf2.utils.gradient(u_x, x)
    v_xx = pinnstf2.utils.gradient(v_x, x)

    outputs["f_u"] = u_t + 0.5 * v_xx + (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["v"]
    outputs["f_v"] = v_t - 0.5 * u_xx - (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["u"]

    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstf2.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstf2.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf2.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
