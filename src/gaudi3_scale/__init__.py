"""Gaudi 3 Scale Starter - Production Infrastructure for Intel Gaudi 3 HPU Clusters.

This package provides tools and utilities for deploying and managing
large-scale machine learning training on Intel Gaudi 3 accelerators.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .accelerator import GaudiAccelerator
from .optimizer import GaudiOptimizer
from .trainer import GaudiTrainer

__all__ = [
    "GaudiAccelerator",
    "GaudiOptimizer", 
    "GaudiTrainer",
]