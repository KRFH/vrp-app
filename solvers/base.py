"""Base solver interface for CVRPTW
====================================
Abstract base class for CVRPTW solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging

from models import Route
from logging_config import get_logger, log_solver_execution


class CVRPTWSolver(ABC):
    """Abstract base class for CVRPTW solvers."""

    def __init__(self):
        """Initialize solver with empty state."""
        self.data = None
        self.variables = {}
        self.constraints = []
        self.objective = None
        self.logger = get_logger(self.__class__.__name__)

    def solve(self, data: Dict) -> List[Route]:
        """
        Solve CVRPTW problem using the template method pattern.

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
        solver_name = self.get_solver_name()
        log_solver_execution(
            self.logger,
            solver_name,
            "Starting solver execution",
            {
                "num_vehicles": data.get("num_vehicles"),
                "num_nodes": len(data.get("distance_matrix", [])),
                "depot": data.get("depot"),
            },
        )

        self.data = data
        self.variables = {}
        self.constraints = []
        self.objective = None

        # Template method pattern: define the solving process
        log_solver_execution(self.logger, solver_name, "Initializing solver")
        self._initialize_solver()

        log_solver_execution(self.logger, solver_name, "Defining variables")
        self._define_variables()

        log_solver_execution(self.logger, solver_name, "Defining objective")
        self._define_objective()

        log_solver_execution(self.logger, solver_name, "Defining constraints")
        self._define_constraints()

        log_solver_execution(self.logger, solver_name, "Configuring search parameters")
        self._configure_search_parameters()

        log_solver_execution(self.logger, solver_name, "Solving problem")
        solution = self._solve_problem()

        log_solver_execution(self.logger, solver_name, "Extracting solution")
        routes = self._extract_solution(solution)

        log_solver_execution(
            self.logger,
            solver_name,
            "Solver execution completed",
            {
                "num_routes": len(routes),
                "total_distance": sum(r.distance for r in routes),
                "total_load": sum(r.load for r in routes),
            },
        )

        return routes

    @abstractmethod
    def _initialize_solver(self) -> None:
        """Initialize the underlying solver engine."""
        pass

    @abstractmethod
    def _define_variables(self) -> None:
        """Define decision variables for the problem."""
        pass

    @abstractmethod
    def _define_objective(self) -> None:
        """Define the objective function to minimize/maximize."""
        pass

    @abstractmethod
    def _define_constraints(self) -> None:
        """Define all problem constraints."""
        pass

    @abstractmethod
    def _configure_search_parameters(self) -> None:
        """Configure search parameters and strategies."""
        pass

    @abstractmethod
    def _solve_problem(self) -> Any:
        """Execute the solving process and return raw solution."""
        pass

    @abstractmethod
    def _extract_solution(self, raw_solution: Any) -> List[Route]:
        """Extract Route objects from the raw solution."""
        pass

    @abstractmethod
    def get_solver_name(self) -> str:
        """Return the name of this solver."""
        pass
