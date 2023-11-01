#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="pinnstf",
    version="0.0.1",
    description="An implementation of PINNs in TensorFlow v2.",
    author="Reza Akbarian Bafghi",
    author_email="reza.akbarianbafghi@coloardo.edu",
    url="https://github.com/rezaakb/pinns-tf",
    install_requires=["hydra-core", "scipy", "pyDOE", "matplotlib", "rootutils", "rich"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
