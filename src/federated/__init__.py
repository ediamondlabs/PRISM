"""
Federated learning module for V2V anomaly detection.

This module implements asynchronous decentralized federated learning (DFL)
where RL agents control communication timing, neighbor selection, and
representation type.

Components:
- RepresentationExtractor: Extract embeddings and prototypes from models
- AsynchronousAggregator: Aggregate received data with staleness weighting
- DFLProtocol: Manage async communication and aggregation per vehicle
"""

from src.federated.aggregation import AsynchronousAggregator
from src.federated.protocol import DFLProtocol
from src.federated.representations import RepresentationExtractor

__all__ = [
    "RepresentationExtractor",
    "AsynchronousAggregator",
    "DFLProtocol",
]
