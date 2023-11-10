<div align="center">

<img src="http://drive.google.com/uc?export=view&id=18OEs1wMiVqpxRTEudW-FzXwJelsx5eHm" width="400">
</br>
</br>

<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rezaakb/pinns-tf2/blob/main/tutorials/0-Schrodinger.ipynb)

<a href="https://arxiv.org/abs/2311.03626">[Paper]</a> - <a href="https://github.com/rezaakb/pinns-torch">[PyTorch]</a> - <a href="https://github.com/maziarraissi/PINNs">[TensorFlow v1]</a>
</div>

## Description

PINNs-TF2 is a Python package built on the TensorFlow V2 framework. It not only accelerates PINNs implementation but also simplifies user interactions by abstracting complex PDE challenges. We underscore the pivotal role of compilers in PINNs, highlighting their ability to boost performance by up to 119x.

<div align="center">
<img src="http://drive.google.com/uc?export=view&id=1vGb-wuPI1bAEsD_5CKtUPUHq8cC1J32X" width="1000">
</br>
<em>Each subplot corresponds to a problem, with its iteration count displayed at the
top. The logarithmic x-axis shows the speed-up factor w.r.t the original code in TensorFlow v1, and the y-axis illustrates the mean relative error.</em>
</div>
</br>


For more information, please refer to our paper:

<a href="https://arxiv.org/abs/2311.03626">PINNs-TF2: Fast and User-Friendly Physics-Informed Neural Networks in TensorFlow V2.</a> Reza Akbarian Bafghi, and Maziar Raissi. ML4PS, NeurIPS, 2023.

## Installation

PINNs-TF2 requires following dependencies to be installed:

- [TensorFlow](https://www.tensorflow.org/install) >=2.0.0
- [Hydra](https://hydra.cc/docs/intro/) >= 1.3

Then, you can install PINNs-TF2 itself via \[pip\]:

```bash
pip install pinnstf2
```

If you intend to introduce new functionalities or make code modifications, we suggest duplicating the repository and setting up a local installation:

```bash
git clone https://github.com/rezaakb/pinns-tf2
cd pinns-tf2

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install package
pip install -e .
```

## Quick start

Explore a variety of implemented examples within the [examples](examples) folder. To run a specific code, such as the one for the Navier-Stokes PDE, you can use:

```bash
python examples/navier_stokes/train.py
```

You can train the model using a specified configuration, like the one found in [examples/navier_stokes/configs/config.yaml](examples/navier_stokes/configs/config.yaml). Parameters can be overridden directly from the command line. For instance:

```bash
python examples/navier_stokes/train.py trainer.max_epochs=20 n_train=3000
```

To utilize our package, there are two primary options:

- Implement your training structures using Hydra, as illustrated in our provided examples.
- Directly incorporate our package to solve your custom problem.

For a practical guide on directly using our package to solve the Schrödinger PDE in a continuous forward problem, refer to our tutorial here: [tutorials/0-Schrodinger.ipynb](tutorials/0-Schrodinger.ipynb).

## Data

The data located on the server and will be downloaded automatically upon running each example.

## Contributing

As this is the first version of our package, there might be scope for enhancements and bug fixes. We highly value community contributions. If you find any issues, missing features, or unusual behavior during your usage of this library, please feel free to open an issue or submit a pull request on GitHub. For any queries, suggestions, or feedback, please send them to [Reza Akbarian Bafghi](https://www.linkedin.com/in/rezaakbarian/) at [reza.akbarianbafghi@colorado.edu](mailto:reza.akbarianbafghi@colorado.edu).

## License

Distributed under the terms of the \[BSD-3\] license, "pinnstf2" is free and open source software.

## Resources

We employed [this template](https://github.com/ashleve/lightning-hydra-template) to develop the package, drawing from its structure and design principles. For a deeper understanding, we recommend visiting their GitHub repository.

## Citation

```
@inproceedings{Bafghi2023PINNsTF2FA,
  title={PINNs-TF2: Fast and User-Friendly Physics-Informed Neural Networks in TensorFlow V2},
  author={Reza Akbarian Bafghi and Maziar Raissi},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:265043331}
}
```
