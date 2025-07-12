"""Data models for CVRPTW solver
==================================
Contains Node, Route dataclasses and utility functions.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class Node:
    idx: int
    x: float
    y: float
    demand: int
    ready: int  # earliest time (minutes)
    due: int  # latest time (minutes)
    service: int  # service duration (minutes)
    task: str = ""  # required skill/task


@dataclass
class Route:
    vehicle_id: int
    path: List[int]
    arrival_times: List[int]
    distance: int
    load: int


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance matrix between all points."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1).round().astype(int)


def create_data_model(nodes: List[Node], num_veh: int, cap: int) -> Dict:
    """Create data model for CVRPTW solver."""
    return {
        "distance_matrix": euclidean_distance_matrix(np.array([[n.x, n.y] for n in nodes])),
        "demands": [n.demand for n in nodes],
        "ready_times": [n.ready for n in nodes],
        "due_times": [n.due for n in nodes],
        "service_times": [n.service for n in nodes],
        "vehicle_capacities": [cap] * num_veh,
        "num_vehicles": num_veh,
        "depot": 0,
    } 