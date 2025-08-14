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
from dash.dash_table import Format
from dash.dependencies import ALL

from models import Node, Route, create_data_model
from solver_factory import SolverFactory
from worker_assignment import parse_workers, assign_workers
from logging_config import setup_logging, get_logger


def normalize_node_df(rows):
    """Return DataFrame with lat/lon columns, accepting old x/y fields."""
    df = pd.DataFrame(rows)
    if "lat" not in df.columns and "x" in df.columns:
        df = df.rename(columns={"x": "lat", "y": "lon"})
    return df.sort_values("id")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# ログ設定の初期化
logger = setup_logging(level="INFO", log_to_file=True, log_to_console=True)
app_logger = get_logger("VRPApp")

SOLUTION_FILE = "data/solution_cache.json"
try:
    with open(SOLUTION_FILE) as f:
        INITIAL_SOLUTION = json.load(f)
        if "nodes" in INITIAL_SOLUTION:
            df = normalize_node_df(INITIAL_SOLUTION["nodes"])
            INITIAL_SOLUTION["nodes"] = df.to_dict("records")
except FileNotFoundError:
    INITIAL_SOLUTION = None

DEFAULT_ROWS = (
    INITIAL_SOLUTION["nodes"]
    if INITIAL_SOLUTION
    else [
        {
            "id": 0,
            "lat": 35.681236,
            "lon": 139.767125,
            "demand": 0,
            "ready": 0,
            "due": 1440,
            "service": 0,
            "task": "depot",
        },
        {
            "id": 1,
            "lat": 35.689487,
            "lon": 139.691711,
            "demand": 10,
            "ready": 300,
            "due": 720,
            "service": 300,
            "task": "delivery",
        },
        {
            "id": 2,
            "lat": 35.658034,
            "lon": 139.751599,
            "demand": 15,
            "ready": 480,
            "due": 900,
            "service": 300,
            "task": "repair",
        },
        {
            "id": 3,
            "lat": 35.673343,
            "lon": 139.710388,
            "demand": 8,
            "ready": 540,
            "due": 1020,
            "service": 200,
            "task": "delivery",
        },
        {
            "id": 4,
            "lat": 35.652832,
            "lon": 139.839478,
            "demand": 12,
            "ready": 600,
            "due": 1080,
            "service": 200,
            "task": "maintenance",
        },
        {
            "id": 5,
            "lat": 35.601298,
            "lon": 139.579506,
            "demand": 7,
            "ready": 360,
            "due": 840,
            "service": 150,
            "task": "delivery",
        },
        {
            "id": 6,
            "lat": 35.733953,
            "lon": 139.731992,
            "demand": 7,
            "ready": 300,
            "due": 840,
            "service": 150,
            "task": "repair",
        },
        {
            "id": 7,
            "lat": 35.710063,
            "lon": 139.8107,
            "demand": 7,
            "ready": 500,
            "due": 900,
            "service": 150,
            "task": "maintenance",
        },
    ]
)

COLS_CFG = [
    {"id": "id", "name": "id", "type": "numeric"},
    {
        "id": "lat",
        "name": "lat",
        "type": "numeric",
        "format": Format.Format(precision=6, scheme=Format.Scheme.fixed),
    },
    {
        "id": "lon",
        "name": "lon",
        "type": "numeric",
        "format": Format.Format(precision=6, scheme=Format.Scheme.fixed),
    },
    {"id": "demand", "name": "demand", "type": "numeric"},
    {"id": "ready", "name": "ready", "type": "numeric"},
    {"id": "due", "name": "due", "type": "numeric"},
    {"id": "service", "name": "service", "type": "numeric"},
    {"id": "task", "name": "task"},
]

# --------------------------------------------------------------------------------------
# Dash UI
# --------------------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
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
                                dcc.Tab(label="Route", value="route"),
                                dcc.Tab(label="Logs", value="logs"),
                            ],
                        ),
                        dcc.Loading(
                            children=[
                                html.Div(id="tab-output"),
                                dash_table.DataTable(id="summary"),
                            ]
                        ),
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
        rows.append(
            {"id": next_id, "lat": 35.0, "lon": 139.0, "demand": 1, "ready": 0, "due": 1440, "service": 10, "task": ""}
        )
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
    State("tab", "value"),
    prevent_initial_call=True,
)
def compute_solution(_, rows, veh, cap, solver_name, workers, current_tab):
    try:
        app_logger.info(f"Starting solution computation with solver: {solver_name}")
        app_logger.info(f"Parameters: vehicles={veh}, capacity={cap}, workers={workers}")

        # Parse input data
        df = normalize_node_df(rows)
        numeric = ["id", "lat", "lon", "demand", "ready", "due", "service"]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[numeric].isnull().any().any():
            raise ValueError("All numeric fields must have valid numbers")
        if 0 not in df["id"].values:
            raise ValueError("Row with id 0 (depot) is required")
        df = df.sort_values("id")

        app_logger.info(f"Input data parsed: {len(df)} nodes")

        # Create nodes
        nodes = [
            Node(
                int(r.id),
                float(r.lat),
                float(r.lon),
                int(r.demand),
                int(r.ready),
                int(r.due),
                int(r.service),
                str(r.task),
            )
            for r in df.itertuples()
        ]

        # Create solver and solve
        app_logger.info(f"Creating solver: {solver_name}")
        solver = SolverFactory.create_solver(solver_name)

        # Parse workers and create data model
        veh_count = int(veh)
        worker_info = parse_workers(workers or "", veh_count)
        app_logger.info(f"Workers parsed: {len(worker_info)} workers")

        data_model = create_data_model(nodes, worker_info, int(cap))
        app_logger.info("Data model created, starting solver execution")

        routes = solver.solve(data_model)
        app_logger.info(f"Solver completed: {len(routes)} routes generated")

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

        app_logger.info("Solution saved to cache file")
        app_logger.info(f"Solution summary: {len(routes)} routes, total distance: {sum(r.distance for r in routes)}")

        return solution, summary_rows, summary_cols, f"Solved using {solver.get_solver_name()}"

    except Exception as e:
        app_logger.error(f"Error during solution computation: {str(e)}")
        return dash.no_update, [], [], f"Error: {e}"


@app.callback(
    Output("tab-output", "children"),
    Input("tab", "value"),
    Input("solution-store", "data"),
    prevent_initial_call=True,
)
def update_graph(tab, sol):
    if not sol:
        return html.Div(dcc.Graph(figure=go.Figure()))

    df = normalize_node_df(sol["nodes"])
    nodes = [
        Node(r.id, r.lat, r.lon, int(r.demand), int(r.ready), int(r.due), int(r.service), getattr(r, "task", ""))
        for r in df.itertuples()
    ]
    id2node: Dict[int, Node] = {n.idx: n for n in nodes}
    routes = [Route(**r) for r in sol["routes"]]
    worker_info = sol.get("workers", [])
    assignments = sol.get("assignments", {})

    if tab == "map":
        fig = go.Figure()

        # ノード（点）
        fig.add_trace(
            go.Scattermap(
                lat=df.lat,
                lon=df.lon,
                mode="markers+text",
                text=df.id.astype(str),
                textposition="top center",
                marker=dict(size=10),
                name="Nodes",
            )
        )

        # ルート（線＋点）
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
        for r in routes:
            worker = assignments.get(r.vehicle_id)
            if worker is None:
                worker = (
                    worker_info[r.vehicle_id]["name"] if r.vehicle_id < len(worker_info) else f"Worker {r.vehicle_id}"
                )
            lats = [id2node[nid].lat for nid in r.path]
            lons = [id2node[nid].lon for nid in r.path]

            fig.add_trace(
                go.Scattermap(
                    lat=lats,
                    lon=lons,
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(width=3),  # 色は下の update_traces で一括設定も可
                    name=f"{worker} (Veh {r.vehicle_id}, d={r.distance})",
                    # 個別に色を付けたい場合は line=dict(color=palette[r.vehicle_id % len(palette)], width=3)
                )
            )

        # MapLibre 用レイアウト（mapbox_* ではなく map を使う）
        fig.update_layout(
            map=dict(
                style="open-street-map",  # 無料スタイル
                zoom=10,
                center=dict(lat=35.681236, lon=139.767125),
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        # ルートごとに色を当てたいとき（任意）:
        for i, r in enumerate(routes, start=1):  # 0番目はNodesなので start=1
            fig.data[i].line.color = palette[r.vehicle_id % len(palette)]

        return html.Div(dcc.Graph(figure=fig))

    elif tab == "gantt":
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

        return html.Div(dcc.Graph(figure=fig))

    elif tab == "route":
        # ★ sol から毎回 df を作る
        df = normalize_node_df(sol["nodes"])
        options = [{"label": f"Node {int(n['id'])}", "value": int(n["id"])} for n in sol["nodes"]]

        fig = go.Figure()
        if not df.empty:
            fig.add_trace(
                go.Scattermap(
                    lat=df.lat,
                    lon=df.lon,
                    mode="markers+text",
                    text=df.id.astype(str),  # ★ 表示は文字列
                    textposition="top center",
                    marker=dict(size=10),
                    name="Nodes",
                )
            )

        # 中心はデータの平均に寄せると親切（任意）
        center_lat = float(df.lat.mean()) if not df.empty else 35.681236
        center_lon = float(df.lon.mean()) if not df.empty else 139.767125

        fig.update_layout(
            map=dict(
                style="open-street-map",
                zoom=10,
                center=dict(lat=center_lat, lon=center_lon),
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=500,
        )

        first_dropdown = html.Div(
            dcc.Dropdown(
                id={"type": "route-select", "index": 0},
                options=options,
                multi=True,
                placeholder="Select nodes in order",
            ),
            style={"marginBottom": "0.5rem"},
        )

        return html.Div(
            [
                html.Div(id="route-dropdown-container", children=[first_dropdown]),
                dbc.Button("Add Route", id="add-route", size="sm", color="secondary", className="mb-2"),
                dcc.Graph(id="route-graph", figure=fig),
            ]
        )

    elif tab == "logs":
        try:
            # ログファイルを読み込む
            from logging_config import LOG_FILE
            import os

            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    log_content = f.read()
            else:
                log_content = "ログファイルがまだ作成されていません。"

            # ログ内容をHTMLで表示
            return html.Div(
                [
                    html.H4("実行ログ", className="mb-3"),
                    html.Div(
                        [
                            html.Pre(
                                log_content,
                                style={
                                    "fontFamily": "monospace",
                                    "fontSize": "12px",
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #dee2e6",
                                    "borderRadius": "4px",
                                    "padding": "15px",
                                    "overflow": "auto",
                                    "maxHeight": "500px",
                                    "whiteSpace": "pre-wrap",
                                    "wordWrap": "break-word",
                                },
                            )
                        ]
                    ),
                ]
            )

        except Exception as e:
            app_logger.error(f"Error reading log file: {str(e)}")
            return html.Div(
                [
                    html.H4("ログ表示エラー", className="text-danger"),
                    html.P(f"ログファイルの読み込みエラー: {str(e)}", className="text-danger"),
                ]
            )


@app.callback(
    Output("route-dropdown-container", "children"),
    Input("add-route", "n_clicks"),
    State("route-dropdown-container", "children"),
    State("solution-store", "data"),
    prevent_initial_call=True,
)
def add_route_dropdown(n_clicks, children, sol):
    if not sol:
        return children
    options = [{"label": f"Node {n['id']}", "value": int(n["id"])} for n in sol["nodes"]]
    children = children or []
    idx = len(children)
    children.append(
        html.Div(
            dcc.Dropdown(
                id={"type": "route-select", "index": idx},
                options=options,
                multi=True,
                placeholder="Select nodes in order",
            ),
            style={"marginBottom": "0.5rem"},
        )
    )
    return children


@app.callback(
    Output("route-graph", "figure"),
    Input({"type": "route-select", "index": ALL}, "value"),
    State("solution-store", "data"),
    prevent_initial_call=True,
)
def update_routes(selected_nodes_list, sol):
    if not sol:
        return go.Figure()

    df = normalize_node_df(sol["nodes"])
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(
            go.Scattermap(
                lat=df.lat,
                lon=df.lon,
                mode="markers+text",
                text=df.id.astype(str),  # ★ 表示は文字列
                textposition="top center",
                marker=dict(size=10),
                name="Nodes",
            )
        )

    # ★ id を int キーに（Dropdown values と型を合わせる）
    id2node = df.set_index(df["id"].astype(int)).to_dict("index")

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe
    for idx, selected in enumerate(selected_nodes_list or []):
        if selected and len(selected) >= 2:
            # ★ 値は int 前提（念のためキャスト）
            seq = [int(nid) for nid in selected if int(nid) in id2node]
            if len(seq) >= 2:
                lats = [id2node[n]["lat"] for n in seq]
                lons = [id2node[n]["lon"] for n in seq]
                fig.add_trace(
                    go.Scattermap(
                        lat=lats,
                        lon=lons,
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(width=2, color=palette[idx % len(palette)]),
                        name=f"Route {idx + 1}",
                    )
                )

    center_lat = float(df.lat.mean()) if not df.empty else 35.681236
    center_lon = float(df.lon.mean()) if not df.empty else 139.767125
    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=10,
            center=dict(lat=center_lat, lon=center_lon),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
    )
    return fig


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True)
