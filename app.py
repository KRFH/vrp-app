"""Dash app for interactive Capacitated Vehicle Routing Problem (CVRP)
====================================================================
This app lets users upload or paste a set of customer nodes (coordinates + demand),
choose the number of vehicles and their capacity, and then solve the CVRP with
Google OR‑Tools.  Results are displayed graphically and in a table.

Run:
    pip install dash==2.15.0 dash-bootstrap-components pandas numpy plotly
    pip install ortools

Then:
    python cvrp_dash_app.py
Navigate to http://127.0.0.1:8050/ in your browser.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, ctx, dcc, html
from dash import dash_table  # type: ignore
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --------------------------------------------------------------------------------------
# Utility dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class Node:
    idx: int
    x: float
    y: float
    demand: int


@dataclass
class Route:
    vehicle_id: int
    path: List[int]
    distance: float
    load: int


# --------------------------------------------------------------------------------------
# Solver helpers
# --------------------------------------------------------------------------------------


def _euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pair‑wise Euclidean distance matrix."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1).round(2)


def _create_data_model(nodes: List[Node], num_vehicles: int, capacity: int):
    coords = np.array([[n.x, n.y] for n in nodes])
    distance_matrix = _euclidean_distance_matrix(coords).astype(int)
    return {
        "distance_matrix": distance_matrix,
        "demands": [n.demand for n in nodes],
        "vehicle_capacities": [capacity] * num_vehicles,
        "num_vehicles": num_vehicles,
        "depot": 0,
    }


def _solve_cvrp(data: dict) -> List[Route]:
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback (edge weights)
    def distance_callback(from_index, to_index):
        return data["distance_matrix"][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Capacity constraint
    def demand_callback(from_index):
        return data["demands"][manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],
        True,  # start cumul to zero
        "Capacity",
    )

    # First solution heuristic + metaheuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise RuntimeError("No feasible solution found with given parameters.")

    routes: List[Route] = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        path = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            path.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            route_load += data["demands"][node_index]
        path.append(data["depot"])  # return to depot
        routes.append(Route(vehicle_id, path, route_distance, route_load))
    return routes


# --------------------------------------------------------------------------------------
# Dash application
# --------------------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CVRP Solver"

SAMPLE_CSV = "id,x,y,demand\n" "0,50,50,0\n" "1,60,20,10\n" "2,95,80,15\n" "3,25,30,8\n" "4,10,70,12\n" "5,80,40,7\n"

app.layout = dbc.Container(
    [
        html.H2("Capacitated Vehicle Routing Problem Interactive Solver"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Input"),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                [
                                    "Drag and Drop or ",
                                    html.A("Select CSV File"),
                                ]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                            },
                            multiple=False,
                        ),
                        html.Br(),
                        dbc.Label("...or edit raw CSV:"),
                        dcc.Textarea(
                            id="raw-csv",
                            value=SAMPLE_CSV,
                            style={"width": "100%", "height": "180px", "fontFamily": "monospace"},
                        ),
                        html.Br(),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Number of vehicles"),
                                dbc.Input(id="num-vehicles", type="number", value=2, min=1),
                            ]
                        ),
                        html.Br(),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Vehicle capacity"),
                                dbc.Input(id="vehicle-capacity", type="number", value=30, min=1),
                            ]
                        ),
                        html.Br(),
                        dbc.Button("Solve", id="solve-btn", color="primary", className="me-2"),
                        html.Span(id="alert", className="text-danger"),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dcc.Loading(
                            id="loading-1",
                            type="default",
                            children=[dcc.Graph(id="graph"), dash_table.DataTable(id="route-table")],
                        )
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


def _parse_csv(contents: str | None, filename: str | None, raw_text: str) -> pd.DataFrame:
    """Return DataFrame from either uploaded file or raw textarea."""
    if contents and filename:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    else:
        df = pd.read_csv(io.StringIO(raw_text))
    required_cols = {"id", "x", "y", "demand"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    df.sort_values("id", inplace=True)
    return df


@app.callback(
    Output("graph", "figure"),
    Output("route-table", "data"),
    Output("route-table", "columns"),
    Output("alert", "children"),
    Input("solve-btn", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("raw-csv", "value"),
    State("num-vehicles", "value"),
    State("vehicle-capacity", "value"),
    prevent_initial_call=True,
)
def solve_cvrp_callback(n_clicks, contents, filename, raw_text, num_vehicles, capacity):  # noqa: N802
    try:
        df = _parse_csv(contents, filename, raw_text)
        nodes = [Node(row.id, row.x, row.y, int(row.demand)) for row in df.itertuples()]
        data = _create_data_model(nodes, int(num_vehicles), int(capacity))
        routes = _solve_cvrp(data)

        # Build figure
        fig = go.Figure()
        # colors = px.colors.qualitative.Plotly
        coords = np.array([[n.x, n.y] for n in nodes])
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+text",
                text=[str(n.idx) for n in nodes],
                textposition="top center",
                marker=dict(size=10, symbol="circle"),
                name="Nodes",
            )
        )
        for r in routes:
            xs = [nodes[idx].x for idx in r.path]
            ys = [nodes[idx].y for idx in r.path]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(width=2),
                    name=f"Vehicle {r.vehicle_id} (dist {r.distance})",
                )
            )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="X",
            yaxis_title="Y",
            legend_title="Routes",
            height=500,
        )

        # Build table
        table_data = [
            {
                "Vehicle": r.vehicle_id,
                "Path": " → ".join(map(str, r.path)),
                "Distance": r.distance,
                "Load": r.load,
            }
            for r in routes
        ]
        columns = [
            {"name": "Vehicle", "id": "Vehicle"},
            {"name": "Path", "id": "Path"},
            {"name": "Distance", "id": "Distance"},
            {"name": "Load", "id": "Load"},
        ]
        return fig, table_data, columns, ""
    except Exception as e:
        return go.Figure(), [], [], f"Error: {e}"


# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
