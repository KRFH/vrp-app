"""Data models for CVRPTW solver
==================================
Contains Node, Route, Worker dataclasses and utility functions.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np


@dataclass
class Worker:
    """Worker with skills information."""

    id: int
    name: str
    skills: Set[str]


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
    required_skills: Set[str] = None  # type: ignore

    def __post_init__(self):
        if self.required_skills is None:
            self.required_skills = {self.task} if self.task else set()


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


def create_data_model(nodes: List[Node], workers: List[Worker], cap: int) -> Dict:
    """Create data model for CVRPTW solver with worker constraints."""
    return {
        "distance_matrix": euclidean_distance_matrix(np.array([[n.x, n.y] for n in nodes])),
        "demands": [n.demand for n in nodes],
        "ready_times": [n.ready for n in nodes],
        "due_times": [n.due for n in nodes],
        "service_times": [n.service for n in nodes],
        "required_skills": [n.required_skills for n in nodes],
        "vehicle_capacities": [cap] * len(workers),
        "num_vehicles": len(workers),
        "workers": workers,
        "depot": 0,
    }
