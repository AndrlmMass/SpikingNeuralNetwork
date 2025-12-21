"""
Setup script for SpikingNeuralNetwork package.
Install in development mode with: pip install -e .
This allows imports to work from anywhere in the project.
"""

from setuptools import setup, find_packages

setup(
    name="spiking-neural-network",
    version="1.0.0",
    description="Spiking Neural Network with Sleep-like Dynamics",
    author="Andreas",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "tqdm",
        "scikit-learn",
        "snntorch",
    ],
)

