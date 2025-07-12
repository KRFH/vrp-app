"""OR-Tools CVRPTW Solver
==========================
Implementation of CVRPTW solver using Google OR-Tools.
"""

from typing import Dict, List

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .base import CVRPTWSolver
from models import Route


class ORToolsSolver(CVRPTWSolver):
    """CVRPTW solver using Google OR-Tools."""
    
    def solve(self, data: Dict) -> List[Route]:
        """
        Solve CVRPTW using OR-Tools.
        
        Args:
            data: Problem data dictionary
            
        Returns:
            List of Route objects
        """
        mgr = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), 
            data["num_vehicles"], 
            data["depot"]
        )
        routing = pywrapcp.RoutingModel(mgr)

        # Cost: distance
        dist_cb_idx = routing.RegisterTransitCallback(
            lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
        )
        routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

        # Time callback = travel + service at origin
        time_cb_idx = routing.RegisterTransitCallback(
            lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
            + data["service_times"][mgr.IndexToNode(i)]
        )

        # Time dimension (with waiting slack)
        routing.AddDimension(
            time_cb_idx,
            1000,  # slack (waiting)
            max(data["due_times"]) + 60,
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Capacity dimension
        demand_cb_idx = routing.RegisterUnaryTransitCallback(
            lambda i: data["demands"][mgr.IndexToNode(i)]
        )
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx, 
            0, 
            data["vehicle_capacities"], 
            True, 
            "Capacity"
        )

        # Apply time-window constraints
        for node_index, (ready, due) in enumerate(zip(data["ready_times"], data["due_times"])):
            time_dim.CumulVar(mgr.NodeToIndex(node_index)).SetRange(ready, due)

        # Search parameters
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(15)

        sol = routing.SolveWithParameters(params)
        if not sol:
            raise RuntimeError("No feasible solution found.")

        routes: List[Route] = []
        for v in range(data["num_vehicles"]):
            idx = routing.Start(v)
            path, arr, dist, load = [], [], 0, 0
            while not routing.IsEnd(idx):
                n = mgr.IndexToNode(idx)
                path.append(n)
                arr.append(sol.Value(time_dim.CumulVar(idx)))
                load += data["demands"][n]
                prev = idx
                idx = sol.Value(routing.NextVar(idx))
                dist += routing.GetArcCostForVehicle(prev, idx, v)
            path.append(data["depot"])
            arr.append(sol.Value(time_dim.CumulVar(idx)))
            routes.append(Route(v, path, arr, dist, load))
        
        return routes
    
    def get_solver_name(self) -> str:
        """Return the name of this solver."""
        return "OR-Tools" 