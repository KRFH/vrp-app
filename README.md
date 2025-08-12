# VRP App

This repository contains a small Dash application that solves the capacitated vehicle routing problem with time windows (CVRPTW).

The interface lets you edit customer nodes in a table, configure the number of vehicles and their capacity, then solve the problem using [OR‑Tools](https://developers.google.com/optimization/).  Results are visualised on a map and as a Gantt chart showing arrival and service times.

## Features

- Editable node table
- Map view of routes and nodes
- 24‑hour Gantt view of each vehicle's schedule
- Custom route visualisation by selecting nodes in order, with support for multiple colour‑coded routes
- Cached solutions saved to `solution_cache.json`
- Comprehensive logging system for solver execution tracking
- Log viewer in the UI to monitor solver performance and execution details

## Running

Install the required packages and run the app:

```bash
pip install dash==2.15.0 dash-bootstrap-components plotly pandas numpy ortools
python app.py
# then browse to http://127.0.0.1:8050/
```

The solution from the previous run is automatically loaded on start so you can continue exploring.

## Logging

The application includes a comprehensive logging system that tracks:
- Solver initialization and configuration
- Variable and constraint definition
- Search parameter configuration
- Solution execution progress
- Final solution statistics

Logs are saved to the `logs/` directory with daily rotation and can be viewed directly in the UI under the "Logs" tab.

The code is released under the terms of the MIT License.
