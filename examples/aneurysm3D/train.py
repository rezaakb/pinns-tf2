from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
import numpy as np
import tensorflow as tf

import pinnstf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import DictConfig


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """

    data = pinnstf.utils.load_data(root_path, "Aneurysm3D.mat")

    t_star = data["t_star"]  # T x 1
    x_star = data["x_star"]  # N x 1
    y_star = data["y_star"]  # N x 1
    z_star = data["z_star"]  # N x 1

    U_star = data["U_star"]  # N x T
    V_star = data["V_star"]  # N x T
    W_star = data["W_star"]  # N x T
    P_star = data["P_star"]  # N x T
    C_star = data["C_star"]  # N x T

    return pinnstf.data.PointCloudData(
        spatial=[x_star, y_star, z_star],
        time=[t_star],
        solution={"u": U_star, "v": V_star, "w": W_star, "p": P_star, "c": C_star},
    )

def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           y: tf.Tensor,
           z: tf.Tensor,
           t: tf.Tensor):   
    """Define the partial differential equations (PDEs).

    :param outputs: Dictionary containing the network outputs for different variables.
    :param x: Spatial coordinate x.
    :param y: Spatial coordinate y.
    :param z: Spatial coordinate z.
    :param t: Temporal coordinate t.
    :param extra_variables: Additional variables if available (optional).
    :return: Dictionary of computed PDE terms for each variable.
    """

    Pec = 1.0 / 0.0101822
    Rey = 1.0 / 0.0101822

    Y = tf.stack([outputs["c"], outputs["u"], outputs["v"], outputs["w"], outputs["p"]], axis=1)    
    shape = tf.shape(Y)
    Y = tf.reshape(Y, [shape[0], -1])
    
    Y_x = pinnstf.utils.fwd_gradient(Y, x)
    Y_y = pinnstf.utils.fwd_gradient(Y, y)
    Y_z = pinnstf.utils.fwd_gradient(Y, z)
    Y_t = pinnstf.utils.fwd_gradient(Y, t)

    Y_xx = pinnstf.utils.fwd_gradient(Y_x, x)
    Y_yy = pinnstf.utils.fwd_gradient(Y_y, y)
    Y_zz = pinnstf.utils.fwd_gradient(Y_z, z)

    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    w_t = Y_t[:,3:4]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]

    outputs["e1"] = c_t + (u * c_x + v * c_y + w * c_z) - (1.0 / Pec) * (c_xx + c_yy + c_zz)
    outputs["e2"] = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Rey) * (u_xx + u_yy + u_zz)
    outputs["e3"] = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Rey) * (v_xx + v_yy + v_zz)
    outputs["e4"] = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Rey) * (w_xx + w_yy + w_zz)
    outputs["e5"] = u_x + v_y + w_z

    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstf.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstf.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
