#!/usr/bin/env python

from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pinnstf2",
    version="0.1.1",
    description="An implementation of PINNs in TensorFlow v2.",
    author="Reza Akbarian Bafghi",
    author_email="reza.akbarianbafghi@coloardo.edu",
    url="https://github.com/rezaakb/pinns-tf2",
    long_description=long_description,
    license='BSD-3-Clause',
    long_description_content_type='text/markdown',
    install_requires=["hydra-core", "scipy", "pyDOE", "matplotlib", "rootutils", "rich", "tqdm"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = pinnstf2.train:main",
        ]
    },
    include_package_data=True,
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
],
)
