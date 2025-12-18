"""
Models module: Neural network models and dynamics.

Contains:
- snn: Main spiking neural network class (snn_sleepy)
- layers: Network creation and weight initialization
- dynamics: Training loop and weight update functions
"""

from .snn import snn_sleepy
from .dynamics import train_network

__all__ = ['snn_sleepy', 'train_network']





