"""Base solver interface for CVRPTW
====================================
Abstract base class for CVRPTW solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from models import Route


class CVRPTWSolver(ABC):
    """Abstract base class for CVRPTW solvers."""
    
    @abstractmethod
    def solve(self, data: Dict) -> List[Route]:
        """
        Solve CVRPTW problem.
        
        Args:
            data: Dictionary containing problem data with keys:
                - distance_matrix: 2D array of distances
                - demands: List of demands for each node
                - ready_times: List of earliest arrival times
                - due_times: List of latest arrival times
                - service_times: List of service durations
                - vehicle_capacities: List of vehicle capacities
                - num_vehicles: Number of vehicles
                - depot: Depot node index
                
        Returns:
            List of Route objects representing the solution
        """
        pass
    
    @abstractmethod
    def get_solver_name(self) -> str:
        """Return the name of this solver."""
        pass 