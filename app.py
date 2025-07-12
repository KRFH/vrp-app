"""Dash app for interactive CVRPTW with Gantt Visualization
===========================================================
Modular version with pluggable solvers.
"""

from __future__ import annotations

import json
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash import dash_table  # type: ignore

from models import Node, Route, create_data_model
from solver_factory import SolverFactory
from worker_assignment import parse_workers, assign_workers

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

SOLUTION_FILE = "data/solution_cache.json"
try:
    with open(SOLUTION_FILE) as f:
        INITIAL_SOLUTION = json.load(f)
except FileNotFoundError:
    INITIAL_SOLUTION = None

DEFAULT_ROWS = (
    INITIAL_SOLUTION["nodes"]
    if INITIAL_SOLUTION
    else [
        {"id": 0, "x": 50, "y": 50, "demand": 0, "ready": 0, "due": 1440, "service": 0, "task": "depot"},
        {"id": 1, "x": 60, "y": 20, "demand": 10, "ready": 300, "due": 720, "service": 300, "task": "delivery"},
        {"id": 2, "x": 95, "y": 80, "demand": 15, "ready": 480, "due": 900, "service": 300, "task": "repair"},
        {"id": 3, "x": 25, "y": 30, "demand": 8, "ready": 540, "due": 1020, "service": 200, "task": "delivery"},
        {"id": 4, "x": 10, "y": 70, "demand": 12, "ready": 600, "due": 1080, "service": 200, "task": "maintenance"},
        {"id": 5, "x": 80, "y": 40, "demand": 7, "ready": 360, "due": 840, "service": 150, "task": "delivery"},
        {"id": 6, "x": 10, "y": 10, "demand": 7, "ready": 300, "due": 840, "service": 150, "task": "repair"},
        {"id": 7, "x": 50, "y": 80, "demand": 7, "ready": 500, "due": 900, "service": 150, "task": "maintenance"},
    ]
)

COLS_CFG = [{"id": c, "name": c} for c in ["id", "x", "y", "demand", "ready", "due", "service", "task"]]

# --------------------------------------------------------------------------------------
# Dash UI
# --------------------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CVRPTW Solver (Modular)"

app.layout = dbc.Container(
    [
        html.H2("CVRPTW – Modular Solver with 24‑hour Gantt"),
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
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Solver"),
                                dcc.Dropdown(
                                    id="solver-select",
                                    options=[
                                        {"label": name, "value": name} for name in SolverFactory.get_available_solvers()
                                    ],
                                    value="ortools",
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Br(),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Workers"),
                                dbc.Input(
                                    id="workers",
                                    type="text",
                                    placeholder="Name:skill1|skill2, ...",
                                    value="A:delivery, B:repair, C:maintenance",
                                ),
                            ]
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
                        dcc.Store(id="solution-store", data=INITIAL_SOLUTION),
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
        rows.append({"id": next_id, "x": 0, "y": 0, "demand": 1, "ready": 0, "due": 1440, "service": 10, "task": ""})
    return rows


@app.callback(
    Output("solution-store", "data"),
    Output("summary", "data"),
    Output("summary", "columns"),
    Output("msg", "children"),
    Input("solve", "n_clicks"),
    State("table", "data"),
    State("veh", "value"),
    State("cap", "value"),
    State("solver-select", "value"),
    State("workers", "value"),
    prevent_initial_call=True,
)
def compute_solution(_, rows, veh, cap, solver_name, workers):
    try:
        # Parse input data
        df = pd.DataFrame(rows)
        numeric = ["id", "x", "y", "demand", "ready", "due", "service"]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[numeric].isnull().any().any():
            raise ValueError("All numeric fields must have valid numbers")
        if 0 not in df["id"].values:
            raise ValueError("Row with id 0 (depot) is required")
        df = df.sort_values("id")

        # Create nodes
        nodes = [
            Node(
                int(r.id),
                float(r.x),
                float(r.y),
                int(r.demand),
                int(r.ready),
                int(r.due),
                int(r.service),
                str(r.task),
            )
            for r in df.itertuples()
        ]

        # Create solver and solve
        solver = SolverFactory.create_solver(solver_name)

        # Parse workers and create data model
        veh_count = int(veh)
        worker_info = parse_workers(workers or "", veh_count)
        data_model = create_data_model(nodes, worker_info, int(cap))
        routes = solver.solve(data_model)

        # Assign workers (for display purposes)
        id2node = {n.idx: n for n in nodes}
        assignments = assign_workers(routes, id2node, worker_info)

        worker_list = [w.name for w in worker_info]

        # Create summary
        summary_rows = []
        for r in routes:
            required = {id2node[nid].task for nid in r.path if nid != 0 and id2node[nid].task}
            summary_rows.append(
                {
                    "Vehicle": r.vehicle_id,
                    "Worker": assignments.get(r.vehicle_id, worker_list[r.vehicle_id]),
                    "Tasks": ", ".join(sorted(required)),
                    "Path(arrival)": " → ".join(f"{nid}({arr})" for nid, arr in zip(r.path, r.arrival_times)),
                    "Distance": r.distance,
                    "Load": r.load,
                }
            )
        summary_cols = [
            {"name": c, "id": c} for c in ["Vehicle", "Worker", "Tasks", "Path(arrival)", "Distance", "Load"]
        ]

        # Create solution object
        solution = {
            "nodes": df.to_dict("records"),
            "veh": int(veh),
            "cap": int(cap),
            "solver": solver_name,
            "workers": [{"name": w.name, "skills": sorted(list(w.skills))} for w in worker_info],
            "assignments": assignments,
            "routes": [
                {
                    "vehicle_id": r.vehicle_id,
                    "path": r.path,
                    "arrival_times": r.arrival_times,
                    "distance": r.distance,
                    "load": r.load,
                }
                for r in routes
            ],
        }

        # Save solution
        with open(SOLUTION_FILE, "w") as f:
            json.dump(solution, f)

        return solution, summary_rows, summary_cols, f"Solved using {solver.get_solver_name()}"

    except Exception as e:
        return dash.no_update, [], [], f"Error: {e}"


@app.callback(
    Output("graph", "figure"),
    Input("tab", "value"),
    Input("solution-store", "data"),
    prevent_initial_call=True,
)
def update_graph(tab, sol):
    if not sol:
        return go.Figure()

    df = pd.DataFrame(sol["nodes"]).sort_values("id")
    nodes = [
        Node(r.id, r.x, r.y, int(r.demand), int(r.ready), int(r.due), int(r.service), getattr(r, "task", ""))
        for r in df.itertuples()
    ]
    id2node: Dict[int, Node] = {n.idx: n for n in nodes}
    routes = [Route(**r) for r in sol["routes"]]
    worker_info = sol.get("workers", [])
    assignments = sol.get("assignments", {})

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
            worker = assignments.get(r.vehicle_id)
            if worker is None:
                worker = (
                    worker_info[r.vehicle_id]["name"] if r.vehicle_id < len(worker_info) else f"Worker {r.vehicle_id}"
                )
            xs = [id2node[nid].x for nid in r.path]
            ys = [id2node[nid].y for nid in r.path]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(width=2, color=palette[r.vehicle_id % len(palette)]),
                    name=f"{worker} (Veh {r.vehicle_id}, d={r.distance})",
                )
            )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title="X", yaxis_title="Y", height=500)
    else:
        gantt_data = []
        for r in routes:
            for idx, nid in enumerate(r.path[:-1]):
                if nid == 0:
                    continue
                start = r.arrival_times[idx] / 60.0
                finish = (r.arrival_times[idx] + id2node[nid].service) / 60.0
                worker = assignments.get(r.vehicle_id)
                if worker is None:
                    worker = (
                        worker_info[r.vehicle_id]["name"]
                        if r.vehicle_id < len(worker_info)
                        else f"Worker {r.vehicle_id}"
                    )
                gantt_data.append(
                    {
                        "Worker": worker,
                        "Task": f"Node {nid}",
                        "Start": start,
                        "Finish": finish,
                    }
                )
        if gantt_data:
            fig = go.Figure()
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
            color_idx = 0
            for row in gantt_data:
                fig.add_trace(
                    go.Bar(
                        x=[
                            row["Finish"] - row["Start"],
                        ],
                        base=[
                            row["Start"],
                        ],
                        y=[
                            row["Worker"],
                        ],
                        orientation="h",
                        name=row["Task"],
                        marker_color=palette[color_idx % len(palette)],
                        text=row["Task"],
                        textposition="inside",
                    )
                )
                color_idx += 1
            fig.update_layout(
                xaxis=dict(range=[0, 24], title="Hour of day"),
                yaxis_title="Worker",
                height=500,
                barmode="overlay",
            )
            fig.update_yaxes(autorange="reversed")
        else:
            fig = go.Figure()

    return fig


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
