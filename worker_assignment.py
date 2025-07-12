"""Worker assignment logic for CVRPTW
======================================
Handles parsing worker information and assigning workers to routes based on skills.
"""

from typing import Dict, List

from models import Node, Route


def parse_workers(text: str, veh_count: int) -> List[dict]:
    """
    Parse worker information from text input.
    
    Args:
        text: Comma-separated worker information
        veh_count: Number of vehicles
        
    Returns:
        List of worker dictionaries with name and skills
    """
    workers = []
    if text:
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                name, skill_str = part.split(':', 1)
                skills = {s.strip() for s in skill_str.split('|') if s.strip()}
            else:
                name = part
                skills = set()
            workers.append({'name': name.strip(), 'skills': skills})
    
    # Fill remaining slots with default workers
    while len(workers) < veh_count:
        workers.append({'name': f"Worker {len(workers)}", 'skills': set()})
    
    return workers[:veh_count]


def assign_workers(routes: List[Route], nodes: Dict[int, Node], workers: List[dict]) -> Dict[int, str]:
    """
    Assign workers to routes based on required skills.
    
    Args:
        routes: List of routes to assign workers to
        nodes: Dictionary mapping node ID to Node object
        workers: List of available workers with their skills
        
    Returns:
        Dictionary mapping vehicle_id to assigned worker name
    """
    remaining = workers.copy()
    assignments: Dict[int, str] = {}
    
    for r in routes:
        # Get required skills for this route
        required = {nodes[nid].task for nid in r.path if nid != 0 and nodes[nid].task}
        
        # Find worker with matching skills
        idx = next((i for i, w in enumerate(remaining) if required.issubset(w['skills'])), None)
        
        if idx is None:
            # No worker with required skills, take first available
            idx = 0 if remaining else None
            
        if idx is not None:
            worker = remaining.pop(idx)
        else:
            # No workers available, create default
            worker = {'name': f"Worker {r.vehicle_id}", 'skills': set()}
            
        assignments[r.vehicle_id] = worker['name']
    
    return assignments 