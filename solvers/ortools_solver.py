"""OR-Tools CVRPTW Solver
==========================
Implementation of CVRPTW solver using Google OR-Tools.
"""

from typing import Dict, List, Any, Optional

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .base import CVRPTWSolver
from models import Route
from logging_config import log_solver_execution, log_solution_info


class ORToolsSolver(CVRPTWSolver):
    """CVRPTW solver using Google OR-Tools."""

    def __init__(self):
        """Initialize OR-Tools solver with specific attributes."""
        super().__init__()
        self.mgr: Optional[pywrapcp.RoutingIndexManager] = None
        self.routing: Optional[pywrapcp.RoutingModel] = None
        self.time_dimension: Optional[pywrapcp.RoutingDimension] = None
        self.capacity_dimension: Optional[pywrapcp.RoutingDimension] = None
        self.search_params: Optional[Any] = None

    def _initialize_solver(self) -> None:
        """Initialize OR-Tools routing model."""
        if self.data is None:
            raise ValueError("Data must be set before initializing solver")

        log_solver_execution(
            self.logger,
            self.get_solver_name(),
            "Creating RoutingIndexManager",
            {
                "num_nodes": len(self.data["distance_matrix"]),
                "num_vehicles": self.data["num_vehicles"],
                "depot": self.data["depot"],
            },
        )

        self.mgr = pywrapcp.RoutingIndexManager(
            len(self.data["distance_matrix"]), self.data["num_vehicles"], self.data["depot"]
        )
        self.routing = pywrapcp.RoutingModel(self.mgr)
        self.time_dimension = None
        self.capacity_dimension = None

        log_solver_execution(self.logger, self.get_solver_name(), "Routing model initialized")

    def _define_variables(self) -> None:
        """Define decision variables for routing."""
        if self.routing is None or self.mgr is None or self.data is None:
            raise ValueError("Solver must be initialized before defining variables")

        # Type assertions to help type checker
        routing = self.routing
        mgr = self.mgr
        data = self.data

        log_solver_execution(self.logger, self.get_solver_name(), "Registering callbacks")

        # Distance callback for arc costs
        self.variables["distance_callback"] = routing.RegisterTransitCallback(
            lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
        )

        # Time callback for travel + service time
        self.variables["time_callback"] = routing.RegisterTransitCallback(
            lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
            + data["service_times"][mgr.IndexToNode(i)]
        )

        # Demand callback for capacity
        self.variables["demand_callback"] = routing.RegisterUnaryTransitCallback(
            lambda i: data["demands"][mgr.IndexToNode(i)]
        )

        log_solver_execution(
            self.logger, self.get_solver_name(), "Callbacks registered", {"callbacks": list(self.variables.keys())}
        )

    def _define_objective(self) -> None:
        """Define objective function to minimize total distance."""
        if self.routing is None:
            raise ValueError("Solver must be initialized before defining objective")

        self.routing.SetArcCostEvaluatorOfAllVehicles(self.variables["distance_callback"])
        self.objective = "minimize_total_distance"

        log_solver_execution(self.logger, self.get_solver_name(), "Objective defined", {"objective": self.objective})

    def _define_constraints(self) -> None:
        """Define all problem constraints."""
        log_solver_execution(self.logger, self.get_solver_name(), "Defining constraints")

        self._define_time_constraints()
        self._define_capacity_constraints()
        self._define_time_window_constraints()
        self._define_worker_constraints()

        log_solver_execution(
            self.logger, self.get_solver_name(), "Constraints defined", {"constraints": self.constraints}
        )

    def _define_time_constraints(self) -> None:
        """Define time dimension constraints."""
        if self.routing is None or self.data is None:
            raise ValueError("Solver must be initialized before defining constraints")

        self.routing.AddDimension(
            self.variables["time_callback"],
            1000,  # slack (waiting time)
            max(self.data["due_times"]) + 60,  # maximum time
            False,  # start cumul to zero
            "Time",
        )
        self.time_dimension = self.routing.GetDimensionOrDie("Time")
        self.constraints.append("time_dimension")

    def _define_capacity_constraints(self) -> None:
        """Define capacity dimension constraints."""
        if self.routing is None or self.data is None:
            raise ValueError("Solver must be initialized before defining constraints")

        self.routing.AddDimensionWithVehicleCapacity(
            self.variables["demand_callback"],
            0,  # null capacity slack
            self.data["vehicle_capacities"],
            True,  # start cumul to zero
            "Capacity",
        )
        self.capacity_dimension = self.routing.GetDimensionOrDie("Capacity")
        self.constraints.append("capacity_dimension")

    def _define_time_window_constraints(self) -> None:
        """Define time window constraints for each node."""
        if self.time_dimension is None or self.mgr is None or self.data is None:
            raise ValueError("Time dimension must be defined before time window constraints")

        for node_index, (ready, due) in enumerate(zip(self.data["ready_times"], self.data["due_times"])):
            self.time_dimension.CumulVar(self.mgr.NodeToIndex(node_index)).SetRange(ready, due)
        self.constraints.append("time_windows")

    def _define_worker_constraints(self) -> None:
        """Define worker skill constraints using SetAllowedVehiclesForIndex."""
        if self.routing is None or self.mgr is None or self.data is None:
            raise ValueError("Solver must be initialized before defining constraints")

        # Get worker skills and node requirements
        workers = self.data.get("workers", [])
        required_skills = self.data.get("required_skills", [])

        if not workers or not required_skills:
            return  # No worker constraints to apply

        # For each node, determine which vehicles (workers) can visit it
        for node_idx, node_skills in enumerate(required_skills):
            if not node_skills:  # No skills required, all vehicles can visit
                continue

            # Find vehicles (workers) that have the required skills
            allowed_vehicles = []
            for vehicle_id, worker in enumerate(workers):
                if node_skills.issubset(worker.skills):
                    allowed_vehicles.append(vehicle_id)

            # If no vehicles can handle this node, skip it (will be handled by solver)
            if allowed_vehicles:
                routing_idx = self.mgr.NodeToIndex(node_idx)
                self.routing.SetAllowedVehiclesForIndex(allowed_vehicles, routing_idx)

        self.constraints.append("worker_skills")

    def _configure_search_parameters(self) -> None:
        """Configure search parameters and strategies."""
        self.search_params = pywrapcp.DefaultRoutingSearchParameters()
        params = self.search_params
        # type: ignore
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(15)

        log_solver_execution(
            self.logger,
            self.get_solver_name(),
            "Search parameters configured",
            {
                "first_solution_strategy": "PATH_CHEAPEST_ARC",
                "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_seconds": 15,
            },
        )

    def _solve_problem(self) -> Any:
        """Execute the solving process and return raw solution."""
        if self.routing is None or self.search_params is None:
            raise ValueError("Solver must be configured before solving")

        log_solver_execution(self.logger, self.get_solver_name(), "Starting OR-Tools solve")

        solution = self.routing.SolveWithParameters(self.search_params)
        if not solution:
            self.logger.error(f"[{self.get_solver_name()}] No feasible solution found")
            raise RuntimeError("No feasible solution found.")

        log_solver_execution(
            self.logger, self.get_solver_name(), "OR-Tools solve completed", {"solution_status": "FEASIBLE"}
        )
        return solution

    def _extract_solution(self, raw_solution: Any) -> List[Route]:
        """Extract Route objects from the raw solution."""
        if self.routing is None or self.mgr is None or self.time_dimension is None or self.data is None:
            raise ValueError("Solver must be properly initialized before extracting solution")

        routes: List[Route] = []

        for vehicle_id in range(self.data["num_vehicles"]):
            idx = self.routing.Start(vehicle_id)
            path, arrival_times, distance, load = [], [], 0, 0

            while not self.routing.IsEnd(idx):
                node = self.mgr.IndexToNode(idx)
                path.append(node)
                arrival_times.append(raw_solution.Value(self.time_dimension.CumulVar(idx)))
                load += self.data["demands"][node]

                prev_idx = idx
                idx = raw_solution.Value(self.routing.NextVar(idx))
                distance += self.routing.GetArcCostForVehicle(prev_idx, idx, vehicle_id)

            # Add depot to end of path
            path.append(self.data["depot"])
            arrival_times.append(raw_solution.Value(self.time_dimension.CumulVar(idx)))

            routes.append(Route(vehicle_id, path, arrival_times, distance, load))

        return routes

    def get_solver_name(self) -> str:
        """Return the name of this solver."""
        return "OR-Tools"

    def get_solution_info(self) -> Dict[str, Any]:
        """Get information about the solution and constraints."""
        solution_info = {
            "solver_name": self.get_solver_name(),
            "objective": self.objective,
            "constraints": self.constraints,
            "variables": list(self.variables.keys()),
            "search_strategy": "PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH",
            "time_limit_seconds": 15,
        }

        # ソリューション情報をログに記録
        log_solution_info(self.logger, self.get_solver_name(), solution_info)

        return solution_info
