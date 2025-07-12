"""Solver factory for CVRPTW
============================
Factory class for creating and managing different CVRPTW solvers.
"""

from typing import Dict, Type, List

from solvers.base import CVRPTWSolver
from solvers.ortools_solver import ORToolsSolver
from solvers.pulp_solver import PuLPSolver


class SolverFactory:
    """Factory for creating CVRPTW solvers."""
    
    _solvers: Dict[str, Type[CVRPTWSolver]] = {
        "ortools": ORToolsSolver,
        "pulp": PuLPSolver,
    }
    
    @classmethod
    def get_available_solvers(cls) -> List[str]:
        """Get list of available solver names."""
        return list(cls._solvers.keys())
    
    @classmethod
    def create_solver(cls, solver_name: str) -> CVRPTWSolver:
        """
        Create a solver instance by name.
        
        Args:
            solver_name: Name of the solver to create
            
        Returns:
            Solver instance
            
        Raises:
            ValueError: If solver name is not recognized
        """
        if solver_name not in cls._solvers:
            available = ", ".join(cls.get_available_solvers())
            raise ValueError(f"Unknown solver '{solver_name}'. Available: {available}")
        
        return cls._solvers[solver_name]()
    
    @classmethod
    def register_solver(cls, name: str, solver_class: Type[CVRPTWSolver]) -> None:
        """
        Register a new solver class.
        
        Args:
            name: Name for the solver
            solver_class: Solver class to register
        """
        cls._solvers[name] = solver_class 