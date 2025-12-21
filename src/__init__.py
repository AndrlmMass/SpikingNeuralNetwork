"""
Spiking Neural Network with Sleep-like Dynamics
================================================

A spiking neural network implementation with biologically-inspired
sleep-like homeostatic mechanisms for weight regulation.

Main Components
---------------
- snn_sleepy : Main network class
- train_network : Core training function

Quick Start
-----------
>>> from src.models.snn import snn_sleepy
>>> snn = snn_sleepy()
>>> snn.prepare_data(image_dataset="geomfig")
>>> snn.prepare_network()
>>> snn.train_network(train_weights=True, sleep=True)
"""

__version__ = "1.0.0"
__author__ = "Andreas"

# Core imports (only the most essential)
from .models.SNN_sleepy.snn import snn_sleepy

__all__ = [
    'snn_sleepy',
]
