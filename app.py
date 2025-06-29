"""Dash app for interactive CVRPTW with Gantt Visualization
===========================================================
Fully working version – fixes truncated code and ensures Gantt tab refreshes
correctly when switched.

Features
--------
* **Editable table** for nodes (no CSV upload)
* **Map** view with routes & nodes
* **Gantt** view on a 24‑hour timeline (arrival + service)
* Recalculates solution on **Solve** click *or* tab switch

Usage
-----
```bash
pip install dash==2.15.0 dash-bootstrap-components plotly pandas numpy ortools
python cvrp_dash_app.py
# → browse to http://127.0.0.1:8050/
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash import dash_table  # type: ignore
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass
class Node:
    idx: int
    x: float
    y: float
    demand: int
    ready: int  # earliest time (minutes)
    due: int  # latest time (minutes)
    service: int  # service duration (minutes)


@dataclass
class Route:
    vehicle_id: int
    path: List[int]
    arrival_times: List[int]
    distance: int
    load: int


# --------------------------------------------------------------------------------------
# OR‑Tools helpers
# --------------------------------------------------------------------------------------


def _euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1).round().astype(int)


def _create_data_model(nodes: List[Node], num_veh: int, cap: int):
    return {
        "distance_matrix": _euclidean_distance_matrix(np.array([[n.x, n.y] for n in nodes])),
        "demands": [n.demand for n in nodes],
        "ready_times": [n.ready for n in nodes],
        "due_times": [n.due for n in nodes],
        "service_times": [n.service for n in nodes],
        "vehicle_capacities": [cap] * num_veh,
        "num_vehicles": num_veh,
        "depot": 0,
    }


def _solve_cvrptw(data: dict) -> List[Route]:
    mgr = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(mgr)

    # cost: distance
    dist_cb_idx = routing.RegisterTransitCallback(
        lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    # time callback = travel + service at origin
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
    demand_cb_idx = routing.RegisterUnaryTransitCallback(lambda i: data["demands"][mgr.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, data["vehicle_capacities"], True, "Capacity")

    # Apply time‑window constraints
    for node_index, (ready, due) in enumerate(zip(data["ready_times"], data["due_times"])):
        time_dim.CumulVar(mgr.NodeToIndex(node_index)).SetRange(ready, due)

    # Search params
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


# --------------------------------------------------------------------------------------
# Dash UI
# --------------------------------------------------------------------------------------

DEFAULT_ROWS = [
    {"id": 0, "x": 50, "y": 50, "demand": 0, "ready": 0, "due": 1440, "service": 0},  # depot
    {"id": 1, "x": 60, "y": 20, "demand": 10, "ready": 300, "due": 720, "service": 30},
    {"id": 2, "x": 95, "y": 80, "demand": 15, "ready": 480, "due": 900, "service": 30},
    {"id": 3, "x": 25, "y": 30, "demand": 8, "ready": 540, "due": 1020, "service": 20},
    {"id": 4, "x": 10, "y": 70, "demand": 12, "ready": 600, "due": 1080, "service": 20},
    {"id": 5, "x": 80, "y": 40, "demand": 7, "ready": 360, "due": 840, "service": 15},
]

COLS_CFG = [{"id": c, "name": c} for c in ["id", "x", "y", "demand", "ready", "due", "service"]]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CVRPTW Solver"

app.layout = dbc.Container(
    [
        html.H2("CVRPTW – Editable Table & 24‑hour Gantt"),
        dbc.Row(
            [
                # -------- left: parameters --------
                dbc.Col(
                    [
                        dash_table.DataTable(
                            id="table",
                            data=DEFAULT_ROWS,
                            columns=COLS_CFG,
                            editable=True,
                            row_deletable=True,
                            style_table={"height": "320px", "overflowY": "auto"},
                        ),
                        html.Br(),
                        dbc.Button("Add Row", id="add", size="sm", color="secondary"),
                        html.Hr(),
                        dbc.InputGroup(
                            [dbc.InputGroupText("# Vehicles"), dbc.Input(id="veh", type="number", value=3, min=1)]
                        ),
                        html.Br(),
                        dbc.InputGroup(
                            [dbc.InputGroupText("Capacity"), dbc.Input(id="cap", type="number", value=35, min=1)]
                        ),
                        html.Br(),
                        dbc.Button("Solve", id="solve", color="primary"),
                        html.Span(id="msg", className="text-danger ms-2"),
                    ],
                    width=4,
                ),
                # -------- right: output --------
                dbc.Col(
                    [
                        dcc.Tabs(
                            id="tab",
                            value="map",
                            children=[
                                dcc.Tab(label="Map", value="map"),
                                dcc.Tab(label="Gantt", value="gantt"),
                            ],
                        ),
                        dcc.Loading(children=[dcc.Graph(id="graph"), dash_table.DataTable(id="summary")]),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------


@app.callback(Output("table", "data"), Input("add", "n_clicks"), State("table", "data"), prevent_initial_call=True)
def add_row(n_clicks, rows):
    if n_clicks:
        next_id = max(r["id"] for r in rows) + 1 if rows else 1
        rows.append({"id": next_id, "x": 0, "y": 0, "demand": 1, "ready": 0, "due": 1440, "service": 10})
    return rows


@app.callback(
    Output("graph", "figure"),
    Output("summary", "data"),
    Output("summary", "columns"),
    Output("msg", "children"),
    Input("solve", "n_clicks"),
    Input("tab", "value"),  # recalc when switching tabs too
    State("table", "data"),
    State("veh", "value"),
    State("cap", "value"),
    prevent_initial_call=True,
)
def update_output(_, tab, rows, veh, cap):  # noqa: D401
    try:
        # --- prepare data ---------------------------------
        df = pd.DataFrame(rows)
        if 0 not in df["id"].values:
            raise ValueError("Row with id 0 (depot) is required")
        df = df.sort_values("id")
        nodes = [Node(r.id, r.x, r.y, int(r.demand), int(r.ready), int(r.due), int(r.service)) for r in df.itertuples()]
        routes = _solve_cvrptw(_create_data_model(nodes, int(veh), int(cap)))
        id2node: Dict[int, Node] = {n.idx: n for n in nodes}

        # --- summary table -------------------------------
        summary_rows = [
            {
                "Vehicle": r.vehicle_id,
                "Path(arrival)": " → ".join(f"{nid}({arr})" for nid, arr in zip(r.path, r.arrival_times)),
                "Distance": r.distance,
                "Load": r.load,
            }
            for r in routes
        ]
        summary_cols = [{"name": c, "id": c} for c in ["Vehicle", "Path(arrival)", "Distance", "Load"]]

        # --- figures ------------------------------------
        if tab == "map":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df.x,
                    y=df.y,
                    mode="markers+text",
                    text=df.id,
                    textposition="top center",
                    marker=dict(size=10),
                    name="Nodes",
                )
            )
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
            for r in routes:
                xs = [id2node[nid].x for nid in r.path]
                ys = [id2node[nid].y for nid in r.path]
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        marker=dict(size=6),
                        line=dict(width=2, color=palette[r.vehicle_id % len(palette)]),
                        name=f"Veh {r.vehicle_id} (d={r.distance})",
                    )
                )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title="X", yaxis_title="Y", height=500)
        else:  # gantt
            gantt_data = []
            for r in routes:
                for idx, nid in enumerate(r.path[:-1]):  # exclude final depot return
                    if nid == 0:
                        continue  # skip depot in gantt
                    start = r.arrival_times[idx] / 60.0
                    finish = (r.arrival_times[idx] + id2node[nid].service) / 60.0
                    gantt_data.append(
                        {
                            "Vehicle": f"Veh {r.vehicle_id}",
                            "Task": f"Node {nid}",
                            "Start": start,
                            "Finish": finish,
                        }
                    )
            if gantt_data:
                gdf = pd.DataFrame(gantt_data)
                fig = px.timeline(gdf, x_start="Start", x_end="Finish", y="Vehicle", color="Task", text="Task")
                fig.update_layout(xaxis=dict(range=[0, 24], title="Hour of day"), yaxis_title="Vehicle", height=500)
                fig.update_yaxes(autorange="reversed")
            else:
                fig = go.Figure()

        return fig, summary_rows, summary_cols, ""

    except Exception as e:
        return go.Figure(), [], [], f"Error: {e}"


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
