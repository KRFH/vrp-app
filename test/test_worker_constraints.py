"""Test worker constraints implementation
========================================
Tests the integration of worker skill constraints in OR-Tools solver.
"""

import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Node, Worker, create_data_model
from solvers.ortools_solver import ORToolsSolver


def test_worker_constraints():
    """Test that worker constraints are properly applied."""

    # Create test nodes with different skill requirements
    nodes = [
        Node(0, 35.681236, 139.767125, 0, 0, 1440, 0, "depot"),  # depot (Tokyo Station)
        Node(1, 35.689487, 139.691711, 10, 300, 720, 300, "delivery", {"delivery"}),
        Node(2, 35.658034, 139.751599, 15, 480, 900, 300, "repair", {"repair"}),
        Node(3, 35.673343, 139.710388, 8, 540, 1020, 200, "delivery", {"delivery"}),
        Node(4, 35.652832, 139.839478, 12, 600, 1080, 200, "maintenance", {"maintenance"}),
        Node(5, 35.701298, 139.579506, 7, 360, 840, 150, "delivery", {"delivery"}),
    ]

    # Create workers with different skills
    workers = [
        Worker(0, "Alice", {"delivery"}),
        Worker(1, "Bob", {"repair"}),
        Worker(2, "Charlie", {"maintenance"}),
    ]

    # Create data model
    data_model = create_data_model(nodes, workers, 35)

    # Create and solve
    solver = ORToolsSolver()
    routes = solver.solve(data_model)

    print("=== Worker Constraints Test ===")
    print(f"Workers: {[w.name for w in workers]}")
    print(f"Worker skills: {[w.skills for w in workers]}")
    print(f"Node requirements: {[n.required_skills for n in nodes]}")
    print(f"Number of routes: {len(routes)}")

    # Check that each route only visits nodes that the assigned worker can handle
    for route in routes:
        worker = workers[route.vehicle_id]
        print(f"\nRoute {route.vehicle_id} (Worker: {worker.name}, Skills: {worker.skills}):")
        print(f"  Path: {route.path}")

        # Check each node in the route
        for node_id in route.path:
            if node_id == 0:  # depot
                continue
            node = nodes[node_id]
            if not node.required_skills.issubset(worker.skills):
                print(f"  ERROR: Node {node_id} requires {node.required_skills} but worker has {worker.skills}")
            else:
                print(f"  ✓ Node {node_id} ({node.required_skills}) - OK")


if __name__ == "__main__":
    test_worker_constraints()
