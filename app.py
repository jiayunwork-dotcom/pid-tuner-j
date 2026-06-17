import base64
import io
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import parse_csv, create_demo_loop
from utils.metrics import compute_metrics
from utils.harris import harris_index, harris_gauge_color, harris_status_text
from utils.oscillation import detect_oscillation
from utils.stiction import detect_stiction
from utils.model_id import identify_fopdt, relay_feedback_identify
from utils.pid_tuning import compute_pid_recommendations, format_tuning_table
from utils.simulation import simulate_closed_loop
from utils.report import generate_report

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True)
app.title = "PID调优分析工具"
server = app.server

COLORS = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "card": "#0f3460",
    "accent": "#e94560",
    "text": "#eee",
    "pv": "#00d2ff",
    "sp": "#ff6b6b",
    "co": "#ffd93d",
    "green": "#27ae60",
    "yellow": "#f39c12",
    "red": "#e74c3c",
    "grid": "#333",
}

app.layout = dbc.Container(fluid=True, style={
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "padding": "0",
}, children=[
    dbc.Navbar(color="#0f3460", dark=True, className="mb-0", children=[
        dbc.NavbarBrand([
            html.I(className="fas fa-chart-line me-2"),
            "工业控制回路性能评估与PID调优分析",
        ], style={"fontSize": "18px", "fontWeight": "bold"}),
        dbc.Nav([
            dbc.NavItem(dbc.Button("加载演示数据", id="btn-demo", color="warning",
                                    size="sm", className="me-2")),
            dbc.NavItem(dbc.Button("导出PDF报告", id="btn-export", color="danger", size="sm")),
        ]),
    ]),

    dbc.Row(className="g-0", children=[
        dbc.Col(id="left-panel", width=3, style={
            "backgroundColor": COLORS["panel"],
            "minHeight": "calc(100vh - 56px)",
            "padding": "10px",
            "overflowY": "auto",
            "borderRight": f"1px solid {COLORS['card']}",
        }, children=[
            html.H6("控制回路列表", className="text-white mb-3"),
            html.Div(id="loop-list-container"),
            html.Hr(className="my-3"),
            html.H6("添加控制回路", className="text-white mb-2"),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt me-1"),
                    "拖拽或点击上传CSV",
                ]),
                style={
                    "width": "100%",
                    "height": "50px",
                    "lineHeight": "50px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "color": "#aaa",
                    "backgroundColor": "#0a1628",
                    "cursor": "pointer",
                },
                multiple=True,
            ),
            html.Div(id="upload-config", className="mt-3", children=[
                dbc.Label("采样周期(秒)", html_for="sampling-period", className="text-light small"),
                dbc.Input(id="sampling-period", type="number", value=1.0, size="sm",
                          className="mb-2"),
                dbc.Label("控制器类型", html_for="controller-type", className="text-light small"),
                dbc.Select(id="controller-type", options=[
                    {"label": "P", "value": "P"},
                    {"label": "PI", "value": "PI"},
                    {"label": "PID", "value": "PID"},
                    {"label": "PD", "value": "PD"},
                ], value="PI", size="sm", className="mb-2"),
                dbc.Label("动作方向", html_for="action-direction", className="text-light small"),
                dbc.Select(id="action-direction", options=[
                    {"label": "反作用", "value": "反作用"},
                    {"label": "正作用", "value": "正作用"},
                ], value="反作用", size="sm", className="mb-2"),
                dbc.Label("工艺区域", html_for="loop-area", className="text-light small"),
                dbc.Input(id="loop-area", type="text", value="", placeholder="如:反应区",
                          size="sm", className="mb-2"),
            ]),
            html.Hr(className="my-3"),
            html.H6("工艺区域筛选", className="text-white mb-2"),
            dbc.Select(id="area-filter-select", options=[
                {"label": "全部区域", "value": "all"},
            ], value="all", size="sm", className="mb-2"),
            html.H6("健康评分排名", className="text-white mb-2"),
            html.Div(id="health-ranking"),
            html.Hr(className="my-3"),
            dbc.Button("回路指标对比", id="btn-compare", color="info", size="sm",
                       className="w-100 mb-2"),
        ]),

        dbc.Col(id="right-panel", width=9, style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "calc(100vh - 56px)",
            "padding": "15px",
            "overflowY": "auto",
        }, children=[
            html.Div(id="main-content", children=[
                html.Div(className="text-center mt-5", children=[
                    html.I(className="fas fa-industry fa-4x mb-3", style={"color": COLORS["accent"]}),
                    html.H4("工业控制回路性能评估与PID调优分析", className="text-white"),
                    html.P("上传DCS历史数据或加载演示数据开始分析", className="text-muted"),
                ]),
            ]),
        ]),
    ]),

    dcc.Store(id="loops-store", data=[]),
    dcc.Store(id="selected-loop", data=-1),
    dcc.Store(id="analysis-store", data={}),
    dcc.Store(id="model-store", data={}),
    dcc.Store(id="tuning-store", data={}),
    dcc.Store(id="time-range-store", data={}),
    dcc.Store(id="annotations-store", data={}),
    dcc.Store(id="annotation-pending", data=None),
    dcc.Store(id="metrics-history-store", data={}),
    dcc.Download(id="download-report"),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("多回路性能指标对比")),
        dbc.ModalBody(id="compare-modal-body"),
        dbc.ModalFooter(
            dbc.Button("关闭", id="btn-close-compare", className="ms-auto", size="sm")
        ),
    ], id="compare-modal", size="xl", scrollable=True),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("性能指标历史记录")),
        dbc.ModalBody(id="history-modal-body"),
        dbc.ModalFooter([
            dbc.Button("清空历史", id="btn-clear-history-modal", color="danger", size="sm", className="me-auto"),
            dbc.Button("关闭", id="btn-close-history", className="ms-auto", size="sm"),
        ]),
    ], id="history-modal", size="xl", scrollable=True, is_open=False),
])


def _loop_card(idx, loop_data, selected):
    health = loop_data.get("health_score", None)
    bg = COLORS["card"] if not selected else COLORS["accent"]
    border = f"2px solid {COLORS['accent']}" if selected else "1px solid #333"
    health_badge = ""
    if health is not None:
        color = COLORS["green"] if health >= 80 else (COLORS["yellow"] if health >= 50 else COLORS["red"])
        health_badge = html.Span(f"{health:.0f}", className="badge ms-auto",
                                  style={"backgroundColor": color, "fontSize": "11px"})

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Strong(loop_data["name"][:20], className="text-white"),
                health_badge,
            ], className="d-flex align-items-center"),
            html.Small(f"{loop_data['controller_type']} | {loop_data['action_direction']} | {loop_data['n_points']}pts",
                       className="text-muted"),
        ],
            style={"padding": "8px 10px", "cursor": "pointer"},
        ),
    ], id={"type": "loop-card", "index": idx},
        style={"backgroundColor": bg, "border": border, "marginBottom": "6px",
               "borderRadius": "6px"},
        className="loop-card-hover",
    )


@ app.callback(
    Output("loops-store", "data"),
    Output("loop-list-container", "children"),
    Output("selected-loop", "data"),
    Input("upload-data", "contents"),
    Input("btn-demo", "n_clicks"),
    State("upload-data", "filename"),
    State("sampling-period", "value"),
    State("controller-type", "value"),
    State("action-direction", "value"),
    State("loop-area", "value"),
    State("loops-store", "data"),
)
def handle_data_upload(contents, demo_clicks, filenames, sampling_period,
                       controller_type, action_direction, area, current_loops):
    triggered = ctx.triggered_id
    new_loops = list(current_loops) if current_loops else []

    if triggered == "btn-demo":
        demos = [
            create_demo_loop("Demo-TIC-101", "反应区", 2000, 1.0),
            create_demo_loop("Demo-FIC-201", "输送区", 1500, 0.5),
            create_demo_loop("Demo-PIC-301", "蒸汽区", 1800, 2.0),
        ]
        for d in demos:
            loop_ser = {
                "name": d["name"],
                "pv": d["pv"].tolist(),
                "sp": d["sp"].tolist(),
                "co": d["co"].tolist(),
                "sampling_period": d["sampling_period"],
                "controller_type": d["controller_type"],
                "action_direction": d["action_direction"],
                "area": d["area"],
                "n_points": d["n_points"],
            }
            new_loops.append(loop_ser)

    elif triggered == "upload-data" and contents:
        if not isinstance(contents, list):
            contents = [contents]
        if not isinstance(filenames, list):
            filenames = [filenames]

        for content, filename in zip(contents, filenames):
            loop_data, error = parse_csv(content, filename, sampling_period,
                                          controller_type, action_direction, area)
            if loop_data:
                loop_ser = {
                    "name": loop_data["name"],
                    "pv": loop_data["pv"].tolist(),
                    "sp": loop_data["sp"].tolist(),
                    "co": loop_data["co"].tolist(),
                    "sampling_period": loop_data["sampling_period"],
                    "controller_type": loop_data["controller_type"],
                    "action_direction": loop_data["action_direction"],
                    "area": loop_data["area"],
                    "n_points": loop_data["n_points"],
                }
                new_loops.append(loop_ser)

    selected = 0 if new_loops else -1
    cards = [_loop_card(i, l, i == selected) for i, l in enumerate(new_loops)]
    return new_loops, cards, selected


@ app.callback(
    Output("selected-loop", "data", allow_duplicate=True),
    Input({"type": "loop-card", "index": ALL}, "n_clicks"),
    State("loops-store", "data"),
    prevent_initial_call=True,
)
def select_loop(clicks, loops):
    if not clicks or not loops:
        return -1
    for i, c in enumerate(clicks):
        if c:
            return i
    return 0


@ app.callback(
    Output("loop-list-container", "children", allow_duplicate=True),
    Input("selected-loop", "data"),
    State("loops-store", "data"),
    prevent_initial_call=True,
)
def update_loop_selection(selected, loops):
    if not loops:
        return []
    return [_loop_card(i, l, i == selected) for i, l in enumerate(loops)]


@app.callback(
    Output("main-content", "children"),
    Output("analysis-store", "data"),
    Output("health-ranking", "children"),
    Input("selected-loop", "data"),
    Input("time-range-store", "data"),
    Input("annotations-store", "data"),
    Input("annotation-pending", "data"),
    Input("metrics-history-store", "data"),
    State("loops-store", "data"),
)
def render_main_content(selected, time_ranges, annotations_store, pending_annotation,
                         metrics_history, loops):
    if not loops or selected < 0 or selected >= len(loops):
        return html.Div(className="text-center mt-5", children=[
            html.I(className="fas fa-industry fa-4x mb-3", style={"color": COLORS["accent"]}),
            html.H4("工业控制回路性能评估与PID调优分析", className="text-white"),
            html.P("上传DCS历史数据或加载演示数据开始分析", className="text-muted"),
        ]), {}, []

    loop = loops[selected]
    pv_full = np.array(loop["pv"])
    sp_full = np.array(loop["sp"])
    co_full = np.array(loop["co"])
    ts = loop["sampling_period"]

    range_key = str(selected)
    selected_range = time_ranges.get(range_key, None) if time_ranges else None

    if selected_range and "x" in selected_range:
        x_range = selected_range["x"]
        t_full = np.arange(len(pv_full)) * ts
        mask = (t_full >= x_range[0]) & (t_full <= x_range[1])
        pv = pv_full[mask]
        sp = sp_full[mask]
        co = co_full[mask]
        range_display = f"已选择: {x_range[0]:.1f}s - {x_range[1]:.1f}s"
    else:
        pv = pv_full
        sp = sp_full
        co = co_full
        range_display = "使用全部数据 (拖拽图表选择分析时间段)"

    if len(pv) < 10:
        return html.Div(className="text-center mt-5", children=[
            html.I(className="fas fa-exclamation-triangle fa-4x mb-3", style={"color": COLORS["yellow"]}),
            html.H4("选择的时间段数据点不足", className="text-white"),
            html.P("请选择更长的时间段", className="text-muted"),
        ]), {}, []

    metrics = compute_metrics(pv, sp, co, ts, loop["controller_type"])

    harris_val, improvement, actual_var = harris_index(pv, sp, ts, loop["controller_type"])

    osc_result = detect_oscillation(pv, sp, co, ts)

    stic_result = detect_stiction(pv, sp, co, ts)

    osc_severity = 0.0
    if osc_result["is_oscillating"]:
        osc_severity = min(100, (1.0 / (osc_result.get("period", 100) / 10 + 0.1)) * 50)
    valve_health = 100.0
    if stic_result["has_stiction"]:
        severity_map = {"轻微": 30, "中度": 60, "严重": 90}
        valve_health = 100 - severity_map.get(stic_result["stiction_severity"], 30)
    harris_norm = harris_val * 100

    health_score = harris_norm * 0.4 + (100 - osc_severity) * 0.3 + valve_health * 0.3

    analysis_data = {
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
        "harris": {"harris": float(harris_val), "improvement": float(improvement), "actual_var": float(actual_var)},
        "oscillation": {
            "is_oscillating": osc_result["is_oscillating"],
            "period": float(osc_result["period"]) if osc_result["period"] else None,
            "frequency": float(osc_result["frequency"]) if osc_result["frequency"] else None,
            "amplitude": float(osc_result["amplitude"]) if osc_result["amplitude"] else None,
            "diagnosis": osc_result["diagnosis"],
        },
        "stiction": {k: v for k, v in stic_result.items() if k not in ("ellipse_score",)},
        "health_score": float(health_score),
    }

    ranking_children = _build_ranking(loops, selected)

    annotations = annotations_store.get(str(selected), []) if annotations_store else []
    pending = pending_annotation if pending_annotation and pending_annotation.get("loop_idx") == selected else None

    history = metrics_history.get(str(selected), []) if metrics_history else []
    last_record = history[-1] if history else None

    harris_segments = _compute_harris_segments(pv, sp, co, ts, loop["controller_type"])

    spectrum_data = _compute_spectrum(pv, ts)

    content = html.Div([
        _section_timeseries(pv_full, sp_full, co_full, ts, loop, selected_range,
                            range_display, annotations, pending),
        html.Hr(className="my-3", style={"borderColor": "#333"}),

        dbc.Row([
            dbc.Col(_section_metrics(metrics, last_record), width=6),
            dbc.Col(_section_radar(metrics), width=6),
        ]),
        html.Hr(className="my-3", style={"borderColor": "#333"}),

        dbc.Row([
            dbc.Col(_section_harris(harris_val, improvement, harris_segments), width=4),
            dbc.Col(_section_oscillation(osc_result, ts, spectrum_data), width=4),
            dbc.Col(_section_stiction(stic_result, pv, sp, co), width=4),
        ]),
        html.Hr(className="my-3", style={"borderColor": "#333"}),

        _section_model_id(loop, pv, sp, co, ts),
        html.Hr(className="my-3", style={"borderColor": "#333"}),

        _section_tuning(loop, ts),
    ])

    return content, analysis_data, ranking_children


def _section_timeseries(pv, sp, co, ts, loop, selected_range=None, range_display="",
                         annotations=None, pending_annotation=None):
    n = len(pv)
    t = np.arange(n) * ts

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pv, name="PV", line=dict(color=COLORS["pv"], width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=sp, name="SP", line=dict(color=COLORS["sp"], width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=co, name="CO", line=dict(color=COLORS["co"], width=1),
                              yaxis="y2"))

    shapes = []
    annotations_list = []

    if selected_range and "x" in selected_range:
        x_range = selected_range["x"]
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=x_range[0],
            y0=0,
            x1=x_range[1],
            y1=1,
            fillcolor="rgba(233, 69, 96, 0.15)",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ))

    if annotations:
        for i, ann in enumerate(annotations):
            x0, x1 = ann["x0"], ann["x1"]
            y0, y1 = ann["y0"], ann["y1"]
            shapes.append(dict(
                type="line",
                xref="x",
                yref="y",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="#ff9f43", width=2.5),
            ))
            shapes.append(dict(
                type="circle",
                xref="x",
                yref="y",
                x0=x0 - 2,
                y0=y0 - (y1 - y0) * 0.02,
                x1=x0 + 2,
                y1=y0 + (y1 - y0) * 0.02,
                fillcolor="#ff9f43",
                line=dict(color="#ff9f43", width=1),
            ))
            shapes.append(dict(
                type="circle",
                xref="x",
                yref="y",
                x0=x1 - 2,
                y0=y1 - (y1 - y0) * 0.02,
                x1=x1 + 2,
                y1=y1 + (y1 - y0) * 0.02,
                fillcolor="#ff9f43",
                line=dict(color="#ff9f43", width=1),
            ))

            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            slope = ann.get("slope", 0)
            duration = ann.get("duration", 0)
            annotations_list.append(dict(
                x=mid_x,
                y=mid_y,
                xref="x",
                yref="y",
                text=f"斜率: {slope:.4f}/s<br>时长: {duration:.1f}s",
                showarrow=False,
                font=dict(color="#ff9f43", size=10),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="#ff9f43",
                borderwidth=1,
                borderpad=3,
                ay=-30,
            ))

    if pending_annotation and "x" in pending_annotation:
        x0 = pending_annotation["x"]
        y0 = pending_annotation["y"]
        shapes.append(dict(
            type="circle",
            xref="x",
            yref="y",
            x0=x0 - 3,
            y0=y0 - (np.max(pv) - np.min(pv)) * 0.02,
            x1=x0 + 3,
            y1=y0 + (np.max(pv) - np.min(pv)) * 0.02,
            fillcolor="#ff6b6b",
            line=dict(color="#ff6b6b", width=2),
        ))

    fig.update_layout(
        title=dict(text=f"{loop['name']} - 时序数据预览", font=dict(color="white", size=14)),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor="#0a1628",
        font=dict(color="#ccc"),
        xaxis=dict(title="时间(s)", gridcolor=COLORS["grid"]),
        yaxis=dict(title="PV / SP", gridcolor=COLORS["grid"]),
        yaxis2=dict(title="CO (%)", overlaying="y", side="right", gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", y=1.12),
        height=300,
        margin=dict(l=50, r=50, t=50, b=40),
        dragmode="select",
        shapes=shapes,
        annotations=annotations_list,
    )

    annotation_buttons = []
    if annotations:
        for i, ann in enumerate(annotations):
            annotation_buttons.append(
                dbc.Button(
                    [html.I(className="fas fa-times me-1"), f"标注{i+1}"],
                    id={"type": "delete-annotation", "index": i},
                    size="sm",
                    color="warning",
                    outline=True,
                    className="me-1 mb-1",
                    style={"fontSize": "11px"},
                )
            )

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.H6("数据预览与时间段选择", className="text-white mb-0"),
                    html.Small("拖拽选择分析时间段 | 点击PV曲线添加趋势标注 | 双击重置", className="text-muted"),
                ]),
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-tag me-1"), "添加标注"],
                        id="btn-add-annotation",
                        size="sm",
                        color="warning",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-eraser me-1"), "清除标注"],
                        id="btn-clear-annotations",
                        size="sm",
                        color="secondary",
                        outline=True,
                        className="mb-2 ms-1",
                    ),
                ]),
            ], className="d-flex justify-content-between align-items-start"),
            html.Div(id="annotation-status", className="small mb-2",
                     style={"color": "#ff9f43"},
                     children=""),
            dcc.Graph(id="timeseries-graph", figure=fig,
                      config={"modeBarButtonsToAdd": ["select2d", "resetScale2d"]}),
            html.Div(id="selected-range-display",
                     className="small mt-1",
                     children=range_display,
                     style={"color": COLORS["accent"] if selected_range else "#6c757d"}),
            html.Div(id="annotation-tags", className="mt-2", children=annotation_buttons),
        ]),
    ], style={"backgroundColor": COLORS["panel"], "marginBottom": "10px"})


def _section_metrics(metrics, last_record=None):
    labels = {
        "IAE": ("IAE 绝对误差积分", ""),
        "ISE": ("ISE 误差平方积分", ""),
        "ITAE": ("ITAE 时间加权绝对误差积分", ""),
        "overshoot_pct": ("过冲量", "%"),
        "settling_time": ("调节时间", "s"),
        "steady_state_error": ("稳态误差", ""),
        "oscillation_period": ("振荡周期", "s"),
        "decay_ratio": ("衰减比", ""),
    }

    table_rows = []
    for key, (label, unit) in labels.items():
        val = metrics.get(key, 0)
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)

        trend_html = html.Td("—", className="text-muted text-center", style={"width": "60px"})
        if last_record and key in last_record.get("metrics", {}):
            prev_val = last_record["metrics"][key]
            if prev_val != 0 and not (isinstance(prev_val, float) and np.isnan(prev_val)):
                change_pct = (val - prev_val) / abs(prev_val) * 100
                if abs(change_pct) > 10:
                    color = COLORS["red"]
                else:
                    color = "#ccc"
                if change_pct > 0.5:
                    trend_icon = "↑"
                elif change_pct < -0.5:
                    trend_icon = "↓"
                else:
                    trend_icon = "→"
                trend_html = html.Td(
                    html.Span(trend_icon, style={"color": color, "fontWeight": "bold", "fontSize": "14px"}),
                    className="text-center",
                    style={"width": "60px"}
                )

        table_rows.append(html.Tr([
            html.Td(label, className="text-light py-1"),
            html.Td(f"{val_str} {unit}", className="text-white py-1"),
            trend_html,
        ]))

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H6("性能指标", className="text-white mb-0"),
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-save me-1"), "记录当前指标"],
                        id="btn-record-metrics",
                        size="sm",
                        color="success",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-history me-1"), "查看历史"],
                        id="btn-view-history",
                        size="sm",
                        color="info",
                        outline=True,
                        className="mb-2 ms-1",
                    ),
                ]),
            ], className="d-flex justify-content-between align-items-start"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("指标", className="text-warning small py-1", style={"fontSize": "11px"}),
                    html.Th("当前值", className="text-warning small py-1 text-right", style={"fontSize": "11px"}),
                    html.Th("趋势", className="text-warning small py-1 text-center", style={"width": "60px", "fontSize": "11px"}),
                ])),
                html.Tbody(table_rows),
            ], className="w-100", style={
                "fontSize": "12px",
                "borderCollapse": "collapse",
            }),
            html.Div(id="record-status", className="small mt-2", style={"color": COLORS["green"]}),
        ]),
    ], style={"backgroundColor": COLORS["panel"], "height": "100%"})


def _section_radar(metrics):
    categories = ["IAE", "ISE", "ITAE", "过冲量", "稳态误差", "衰减比"]
    raw_values = [
        metrics.get("IAE", 0),
        metrics.get("ISE", 0),
        metrics.get("ITAE", 0),
        metrics.get("overshoot_pct", 0),
        abs(metrics.get("steady_state_error", 0)),
        metrics.get("decay_ratio", 0),
    ]

    max_vals = [max(v, 1e-6) for v in raw_values]
    normalized = [min(v / m * 5, 5) for v, m in zip(raw_values, max_vals)]
    normalized = [min(n, 5) for n in normalized]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(233, 69, 96, 0.3)",
        line=dict(color=COLORS["accent"], width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0a1628",
            radialaxis=dict(visible=True, range=[0, 5], gridcolor="#333", linecolor="#333"),
            angularaxis=dict(gridcolor="#333", linecolor="#333"),
        ),
        paper_bgcolor=COLORS["panel"],
        font=dict(color="#ccc", size=10),
        height=300,
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return dbc.Card([
        dbc.CardBody([
            html.H6("指标雷达图", className="text-white mb-2"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]),
    ], style={"backgroundColor": COLORS["panel"], "height": "100%"})


def _section_harris(harris_val, improvement, segments=None):
    gauge_color = harris_gauge_color(harris_val)
    status = harris_status_text(harris_val)

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=harris_val,
        title=dict(text="Harris指标", font=dict(color="white", size=13)),
        delta=dict(reference=1.0, decreasing=dict(color=COLORS["red"]),
                    increasing=dict(color=COLORS["green"])),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color="#ccc")),
            bar=dict(color=gauge_color, thickness=0.3),
            steps=[
                dict(range=[0, 0.5], color="rgba(231, 76, 60, 0.2)"),
                dict(range=[0.5, 0.8], color="rgba(243, 156, 18, 0.2)"),
                dict(range=[0.8, 1.0], color="rgba(39, 174, 96, 0.2)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=0.8),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["panel"],
        font=dict(color="#ccc"),
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    segment_content = html.Div()
    if segments and len(segments) == 4:
        seg_vals = [s["harris"] for s in segments]
        seg_labels = [f"时段{i+1}" for i in range(4)]
        bar_colors = []
        abnormal_reasons = []

        for i, s in enumerate(segments):
            deviation = abs(s["harris"] - harris_val) / max(harris_val, 0.01)
            if deviation > 0.2:
                bar_colors.append(COLORS["red"])
                if s.get("sp_changes", 0) > 3:
                    reason = f"时段{i+1}: SP变化频繁({s['sp_changes']}次)，导致控制性能波动"
                elif s.get("co_saturation_pct", 0) > 20:
                    reason = f"时段{i+1}: CO长时间饱和({s['co_saturation_pct']:.1f}%)，阀门受限"
                elif s.get("pv_std_ratio", 1) > 1.5:
                    reason = f"时段{i+1}: PV波动剧烈，可能存在外部扰动"
                else:
                    reason = f"时段{i+1}: 与整体差异较大，建议进一步分析"
                abnormal_reasons.append(reason)
            else:
                bar_colors.append(COLORS["pv"])

        fig_seg = go.Figure()
        fig_seg.add_trace(go.Bar(
            x=seg_labels,
            y=seg_vals,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in seg_vals],
            textposition="outside",
            textfont=dict(color="#ccc", size=10),
        ))
        fig_seg.add_hline(
            y=harris_val,
            line_dash="dash",
            line_color=COLORS["accent"],
            annotation_text=f"总体: {harris_val:.3f}",
            annotation_position="right",
            annotation_font=dict(color=COLORS["accent"], size=10),
        )
        fig_seg.update_layout(
            title=dict(text="分时段Harris指标对比", font=dict(color="white", size=12)),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0a1628",
            font=dict(color="#ccc", size=10),
            height=180,
            margin=dict(l=40, r=20, t=30, b=30),
            yaxis=dict(range=[0, 1.1], gridcolor="#333", title="Harris值"),
            xaxis=dict(gridcolor="#333"),
            showlegend=False,
        )

        reason_html = html.Div()
        if abnormal_reasons:
            reason_items = [html.Li(r, className="small", style={"color": COLORS["red"]})
                            for r in abnormal_reasons]
            reason_html = html.Div([
                html.P("异常分析:", className="small text-warning mb-1 mt-2 fw-bold"),
                html.Ul(reason_items, className="mb-0 ps-3"),
            ])

        segment_content = html.Div([
            html.Hr(className="my-2", style={"borderColor": "#333"}),
            html.H6("分时段对比", className="text-white small mb-2"),
            dcc.Graph(figure=fig_seg, config={"displayModeBar": False}, style={"height": "180px"}),
            reason_html,
        ])

    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.P(status, className="text-center small mb-1",
                   style={"color": gauge_color, "fontWeight": "bold"}),
            html.P(f"潜在方差削减: {improvement:.1f}%", className="text-center text-muted small"),
            segment_content,
        ]),
    ], style={"backgroundColor": COLORS["panel"], "height": "100%"})


def _section_oscillation(osc_result, ts, spectrum_data=None):
    is_osc = osc_result["is_oscillating"]
    diagnosis = osc_result.get("diagnosis", {})

    status_color = COLORS["yellow"] if is_osc else COLORS["green"]
    status_text = "检测到振荡" if is_osc else "未检测到振荡"

    acf = osc_result.get("acf")
    fig_acf = go.Figure()
    if acf is not None and len(acf) > 0:
        fig_acf.add_trace(go.Bar(x=np.arange(len(acf)), y=acf,
                                  marker_color=COLORS["pv"], marker_line_width=0))
        fig_acf.add_hline(y=0, line_color="#666")
    fig_acf.update_layout(
        title="ACF自相关", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        font=dict(color="#ccc", size=9), height=180,
        margin=dict(l=30, r=10, t=30, b=20),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )

    spectrum_content = html.Div()
    if spectrum_data and spectrum_data.get("freqs") is not None:
        freqs = np.array(spectrum_data["freqs"])
        amps = np.array(spectrum_data["amps"])
        peaks = spectrum_data.get("peaks", [])

        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=freqs, y=amps,
            mode="lines",
            line=dict(color=COLORS["pv"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0, 210, 255, 0.2)",
            name="频谱",
        ))

        peak_colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]
        peak_annotations = []
        if peaks and len(peaks) > 0:
            for i, peak in enumerate(peaks[:4]):
                color = peak_colors[i % len(peak_colors)]
                fig_spec.add_vline(
                    x=peak["freq"],
                    line_dash="dash",
                    line_color=color,
                    line_width=1.5,
                )
                peak_annotations.append({
                    "freq": peak["freq"],
                    "period": peak["period"],
                    "amp": peak["amp"],
                    "color": color,
                    "meaning": peak.get("meaning", ""),
                })

        fig_spec.update_layout(
            title=dict(text="频谱分析 (FFT)", font=dict(color="white", size=12)),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0a1628",
            font=dict(color="#ccc", size=9),
            height=180,
            margin=dict(l=40, r=10, t=30, b=30),
            xaxis=dict(title="频率 (Hz)", gridcolor="#333"),
            yaxis=dict(title="幅值", gridcolor="#333", showticklabels=False),
            showlegend=False,
        )

        meaning_list = html.Div()
        if peak_annotations:
            items = []
            for p in peak_annotations:
                freq_str = f"{p['freq']:.4f} Hz"
                period_str = f"({p['period']:.1f}s周期)"
                items.append(html.Li([
                    html.Span("● ", style={"color": p["color"], "fontSize": "12px"}),
                    html.Span(f"{freq_str} {period_str}", className="text-white small"),
                    html.Br(),
                    html.Span(p["meaning"], className="text-muted small"),
                ], className="mb-1"))
            meaning_list = html.Div([
                html.P("主要频率成分:", className="small text-warning mb-1 mt-2 fw-bold"),
                html.Ul(items, className="mb-0 ps-3", style={"listStyle": "none"}),
            ])

        spectrum_content = html.Div([
            html.Hr(className="my-2", style={"borderColor": "#333"}),
            html.H6("频谱分析", className="text-white small mb-2"),
            dcc.Graph(figure=fig_spec, config={"displayModeBar": False}, style={"height": "180px"}),
            meaning_list,
        ])

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span("● ", style={"color": status_color, "fontSize": "16px"}),
                html.Span(status_text, className="text-white small fw-bold"),
            ]),
            dcc.Graph(figure=fig_acf, config={"displayModeBar": False}),
            html.P(f"根因: {diagnosis.get('root_cause', 'N/A')}", className="small text-warning mb-1"),
            html.P(diagnosis.get("description", "")[:100], className="small text-muted"),
            spectrum_content,
        ]),
    ], style={"backgroundColor": COLORS["panel"], "height": "100%"})


def _section_stiction(stic_result, pv, sp, co):
    has_stiction = stic_result["has_stiction"]
    severity = stic_result["stiction_severity"]

    fig = go.Figure()
    step = max(1, len(co) // 500)
    fig.add_trace(go.Scatter(
        x=co[::step], y=(pv - sp)[::step],
        mode="markers", marker=dict(size=2, color=COLORS["co"], opacity=0.5),
        name="CO vs Error",
    ))
    fig.update_layout(
        title="CO-PV相位图(粘滞检测)", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        font=dict(color="#ccc", size=9), height=180,
        margin=dict(l=30, r=10, t=30, b=20),
        xaxis=dict(title="CO", gridcolor="#333"),
        yaxis=dict(title="PV-SP", gridcolor="#333"),
    )

    sev_color = COLORS["red"] if severity == "严重" else (COLORS["yellow"] if severity == "中度" else COLORS["green"])

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span("● ", style={"color": sev_color, "fontSize": "16px"}),
                html.Span(f"阀门粘滞: {severity}", className="text-white small fw-bold"),
            ]),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.P(stic_result.get("recommendation", ""), className="small text-muted"),
        ]),
    ], style={"backgroundColor": COLORS["panel"], "height": "100%"})


def _section_model_id(loop, pv, sp, co, ts):
    return dbc.Card([
        dbc.CardBody([
            html.H6("阶跃响应模型辨识 (FOPDT)", className="text-white mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("辨识方法", className="text-light small"),
                    dbc.Select(id="id-method", options=[
                        {"label": "面积法 (Smith)", "value": "area"},
                        {"label": "优化拟合法", "value": "optimization"},
                        {"label": "继电反馈法", "value": "relay"},
                    ], value="optimization", size="sm", className="mb-2"),
                    dbc.Button("开始辨识", id="btn-identify", color="primary", size="sm",
                                className="me-2"),
                    dbc.Button("自动检测阶跃", id="btn-auto-step", color="secondary", size="sm"),
                    html.Div(id="model-result", className="mt-3"),
                ], width=5),
                dbc.Col([
                    dcc.Graph(id="model-fit-graph", config={"displayModeBar": False},
                              style={"height": "280px"}),
                ], width=7),
            ]),
        ]),
    ], style={"backgroundColor": COLORS["panel"], "marginBottom": "10px"})


def _section_tuning(loop, ts):
    return dbc.Card([
        dbc.CardBody([
            html.H6("PID参数整定建议", className="text-white mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div(id="tuning-table-container"),
                    html.Hr(className="my-2", style={"borderColor": "#333"}),
                    dbc.Label("Lambda闭环时间常数(s)", className="text-light small"),
                    dbc.Input(id="lambda-input", type="number", value=30, size="sm", className="mb-2"),
                    dbc.Label("IMC滤波因子(s)", className="text-light small"),
                    dbc.Input(id="imc-filter-input", type="number", value=10, size="sm", className="mb-2"),
                    dbc.Button("计算整定参数", id="btn-tune", color="success", size="sm",
                                className="me-2"),
                    dbc.Button("仿真对比", id="btn-simulate", color="info", size="sm"),
                ], width=5),
                dbc.Col([
                    dcc.Graph(id="sim-compare-graph", config={"displayModeBar": False},
                              style={"height": "300px"}),
                    html.Div(id="sim-param-select", className="mt-2"),
                ], width=7),
            ]),
        ]),
    ], style={"backgroundColor": COLORS["panel"], "marginBottom": "10px"})


def _build_ranking(loops, selected_idx):
    if not loops:
        return html.P("暂无数据", className="text-muted small")

    scores = []
    for i, loop in enumerate(loops):
        pv = np.array(loop["pv"])
        sp = np.array(loop["sp"])
        ts = loop["sampling_period"]
        try:
            h_val, _, _ = harris_index(pv, sp, ts, loop["controller_type"])
            osc_r = detect_oscillation(pv, sp, np.array(loop["co"]), ts)
            stic_r = detect_stiction(pv, sp, np.array(loop["co"]), ts)

            osc_sev = 0
            if osc_r["is_oscillating"]:
                osc_sev = min(100, 50)
            valve_h = 100
            if stic_r["has_stiction"]:
                sev_map = {"轻微": 30, "中度": 60, "严重": 90}
                valve_h = 100 - sev_map.get(stic_r["stiction_severity"], 30)
            score = h_val * 100 * 0.4 + (100 - osc_sev) * 0.3 + valve_h * 0.3
        except Exception:
            score = 50.0

        scores.append((i, loop["name"], score))

    scores.sort(key=lambda x: x[2])

    items = []
    for idx, name, score in scores:
        color = COLORS["red"] if score < 50 else (COLORS["yellow"] if score < 80 else COLORS["green"])
        badge = "⚠" if score < 50 else ""
        items.append(html.Div([
            html.Span(f"{badge} {name[:15]}", className="text-light small"),
            html.Span(f"{score:.0f}", className="badge ms-auto",
                      style={"backgroundColor": color, "fontSize": "10px"}),
        ], className="d-flex justify-content-between align-items-center py-1 px-2",
            style={"backgroundColor": COLORS["card"] if idx != selected_idx else COLORS["accent"],
                    "borderRadius": "4px", "marginBottom": "3px", "cursor": "pointer"},
            id={"type": "rank-item", "index": idx}))

    return items


@app.callback(
    Output("model-result", "children"),
    Output("model-fit-graph", "figure"),
    Output("model-store", "data"),
    Input("btn-identify", "n_clicks"),
    Input("btn-auto-step", "n_clicks"),
    State("id-method", "value"),
    State("selected-loop", "data"),
    State("loops-store", "data"),
    State("model-store", "data"),
    prevent_initial_call=True,
)
def run_model_identification(identify_clicks, auto_clicks, method, selected, loops, model_data):
    if not loops or selected < 0 or selected >= len(loops):
        return html.P("请先选择回路", className="text-muted small"), go.Figure(), {}

    loop = loops[selected]
    pv = np.array(loop["pv"])
    sp = np.array(loop["sp"])
    co = np.array(loop["co"])
    ts = loop["sampling_period"]

    if method == "relay":
        result, error = relay_feedback_identify(pv, sp, co, ts)
        if error:
            return html.P(f"错误: {error}", className="text-danger small"), go.Figure(), {}

        model_stored = {
            "method": "relay",
            "Ku": result["Ku"],
            "Pu": result["Pu"],
        }

        fig = go.Figure()
        fig.update_layout(
            title="继电反馈辨识", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
            font=dict(color="#ccc"), height=280,
        )
        return html.Div([
            html.P(f"临界增益 Ku = {result['Ku']:.4f}", className="text-white small mb-1"),
            html.P(f"临界周期 Pu = {result['Pu']:.2f} s", className="text-white small"),
        ]), fig, model_stored

    result, error = identify_fopdt(pv, sp, co, ts, method=method)
    if error:
        return html.P(f"错误: {error}", className="text-danger small"), go.Figure(), {}

    model_stored = {
        "K": result["K"],
        "T": result["T"],
        "L": result["L"],
        "r_squared": result["r_squared"],
        "method": result["method"],
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["t_seg"], y=result["pv_seg"],
                              name="实测响应", line=dict(color=COLORS["pv"], width=2)))
    fig.add_trace(go.Scatter(x=result["t_seg"], y=result["model_response"],
                              name="模型响应", line=dict(color=COLORS["sp"], width=2, dash="dash")))
    fig.update_layout(
        title="模型拟合对比", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        font=dict(color="#ccc", size=10), height=280,
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(title="时间(s)", gridcolor="#333"),
        yaxis=dict(title="PV", gridcolor="#333"),
        legend=dict(orientation="h", y=1.12),
    )

    r2 = result["r_squared"]
    if r2 < 0.5:
        r2_color = "text-danger"
        r2_note = " (拟合效果较差，建议使用优化法或检查数据)"
    elif r2 < 0.8:
        r2_color = "text-warning"
        r2_note = " (拟合效果一般)"
    else:
        r2_color = "text-success"
        r2_note = ""

    result_html = html.Div([
        html.P(f"增益 K = {result['K']:.4f}", className="text-white small mb-1"),
        html.P(f"时间常数 T = {result['T']:.2f} s", className="text-white small mb-1"),
        html.P(f"纯滞后 L = {result['L']:.2f} s", className="text-white small mb-1"),
        html.P([
            html.Span(f"R² = {r2:.4f}", className=f"{r2_color} small"),
            html.Span(r2_note, className="text-muted small"),
        ], className="mb-1"),
    ])

    return result_html, fig, model_stored


def _render_tuning_table(model_data, lambda_val, imc_filter, selected, loops):
    if not model_data or model_data.get("method") == "relay":
        return html.P("请先辨识FOPDT模型", className="text-muted small"), {}

    K = model_data.get("K", 0)
    T = model_data.get("T", 0)
    L = model_data.get("L", 0)

    if K == 0 or T == 0:
        return html.P("模型参数无效", className="text-danger small"), {}

    controller_type = loops[selected]["controller_type"] if loops and selected >= 0 else "PID"

    recommendations = compute_pid_recommendations(K, T, L, controller_type, lambda_val, imc_filter)

    rows = format_tuning_table(recommendations)

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("方法", className="text-warning small p-1"),
            html.Th("Kp", className="text-warning small p-1"),
            html.Th("Ki", className="text-warning small p-1"),
            html.Th("Kd", className="text-warning small p-1"),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(r["方法"], className="text-light small p-1"),
                html.Td(r["Kp"], className="text-white small p-1"),
                html.Td(r["Ki"], className="text-white small p-1"),
                html.Td(r["Kd"], className="text-white small p-1"),
            ]) for r in rows
        ]),
    ], className="w-100", style={"fontSize": "11px", "borderCollapse": "collapse",
                                   "border": "1px solid #333"})

    tuning_stored = {k: v for k, v in recommendations.items()}

    return table, tuning_stored


@app.callback(
    Output("tuning-table-container", "children"),
    Output("tuning-store", "data"),
    Input("btn-tune", "n_clicks"),
    Input("model-store", "data"),
    State("lambda-input", "value"),
    State("imc-filter-input", "value"),
    State("selected-loop", "data"),
    State("loops-store", "data"),
    prevent_initial_call=True,
)
def compute_tuning(clicks, model_data, lambda_val, imc_filter, selected, loops):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "model-store" in trigger:
        if not model_data:
            return html.P("请先辨识FOPDT模型", className="text-muted small"), {}
        if model_data.get("method") == "relay":
            return html.P([
                html.I(className="fas fa-info-circle me-2"),
                "继电反馈法已完成，可用于部分整定方法",
            ], className="text-info small"), {}

    return _render_tuning_table(model_data, lambda_val, imc_filter, selected, loops)


@app.callback(
    Output("sim-compare-graph", "figure"),
    Output("sim-param-select", "children"),
    Input("btn-simulate", "n_clicks"),
    State("model-store", "data"),
    State("tuning-store", "data"),
    State("selected-loop", "data"),
    State("loops-store", "data"),
    prevent_initial_call=True,
)
def run_simulation(clicks, model_data, tuning_data, selected, loops):
    if not model_data or not tuning_data:
        return go.Figure(), html.P("请先完成模型辨识和参数整定", className="text-muted small")

    K = model_data.get("K", 0)
    T = model_data.get("T", 0)
    L = model_data.get("L", 0)
    if K == 0 or T == 0:
        return go.Figure(), html.P("模型参数无效", className="text-danger small")

    controller_type = loops[selected]["controller_type"] if loops and selected >= 0 else "PID"
    action_dir = loops[selected]["action_direction"] if loops and selected >= 0 else "反作用"

    first_method = list(tuning_data.values())[0]
    Kp_new = first_method["Kp"]
    Ki_new = first_method["Ki"]
    Kd_new = first_method["Kd"]

    sp_step = 10.0
    pv_range = np.max(np.array(loops[selected]["pv"])) - np.min(np.array(loops[selected]["pv"]))
    if pv_range > 0:
        sp_step = pv_range * 0.2

    t, sp, pv_new, pv_current = simulate_closed_loop(
        K, T, L, Kp_new, Ki_new, Kd_new,
        ts=1.0, sim_time=max(600, T * 10), sp_step_size=sp_step,
        controller_type=controller_type, action_direction=action_dir,
        current_Kp=Kp_new * 0.8, current_Ki=Ki_new * 0.5 if Ki_new else 0,
        current_Kd=Kd_new * 0.5 if Kd_new else 0,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sp, name="SP", line=dict(color=COLORS["sp"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=pv_new, name="新参数响应",
                              line=dict(color=COLORS["green"], width=2)))
    if pv_current is not None:
        fig.add_trace(go.Scatter(x=t, y=pv_current, name="当前参数响应",
                                  line=dict(color=COLORS["pv"], width=1.5, dash="dot")))

    fig.update_layout(
        title="闭环阶跃响应仿真对比", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        font=dict(color="#ccc", size=10), height=300,
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(title="时间(s)", gridcolor="#333"),
        yaxis=dict(title="PV", gridcolor="#333"),
        legend=dict(orientation="h", y=1.12),
    )

    method_names = [v["name"] for v in tuning_data.values()]
    select = html.Div([
        dbc.Label("选择整定方法:", className="text-light small"),
        dbc.Select(id="sim-method-select", options=[
            {"label": n, "value": n} for n in method_names
        ], value=method_names[0] if method_names else None, size="sm"),
    ])

    return fig, select


@app.callback(
    Output("download-report", "data"),
    Input("btn-export", "n_clicks"),
    State("selected-loop", "data"),
    State("loops-store", "data"),
    State("analysis-store", "data"),
    State("model-store", "data"),
    State("tuning-store", "data"),
    prevent_initial_call=True,
)
def export_report(clicks, selected, loops, analysis, model_data, tuning_data):
    if not loops or selected < 0:
        return dash.no_update

    loop = loops[selected]
    loop_data = {
        "name": loop["name"],
        "sampling_period": loop["sampling_period"],
        "controller_type": loop["controller_type"],
        "action_direction": loop["action_direction"],
        "n_points": loop["n_points"],
        "area": loop.get("area", ""),
    }

    metrics = analysis.get("metrics", {})
    harris_result = analysis.get("harris", {})
    oscillation_result = analysis.get("oscillation", {})
    stiction_result = analysis.get("stiction", {})

    buf = generate_report(loop_data, metrics, harris_result, oscillation_result,
                           stiction_result, model_data, tuning_data, None)

    return dcc.send_bytes(buf.getvalue(), f"{loop['name']}_审计报告.pdf")


@app.callback(
    Output("time-range-store", "data"),
    Input("timeseries-graph", "selectedData"),
    State("selected-loop", "data"),
    State("time-range-store", "data"),
    prevent_initial_call=True,
)
def handle_time_range_selection(selected_data, selected_idx, current_ranges):
    if selected_idx < 0:
        return current_ranges or {}

    ranges = dict(current_ranges) if current_ranges else {}
    range_key = str(selected_idx)

    if selected_data and "range" in selected_data:
        ranges[range_key] = selected_data["range"]
    elif selected_data and "x" in selected_data:
        ranges[range_key] = {"x": selected_data["x"], "y": selected_data.get("y")}

    return ranges


@app.callback(
    Output("time-range-store", "data", allow_duplicate=True),
    Input("timeseries-graph", "clickData"),
    State("selected-loop", "data"),
    State("time-range-store", "data"),
    prevent_initial_call=True,
)
def reset_time_range_on_double_click(click_data, selected_idx, current_ranges):
    if not click_data or selected_idx < 0:
        return dash.no_update

    ranges = dict(current_ranges) if current_ranges else {}
    range_key = str(selected_idx)

    points = click_data.get("points", [])
    if len(points) == 0:
        if range_key in ranges:
            del ranges[range_key]
        return ranges

    return dash.no_update


@app.callback(
    Output("sim-compare-graph", "figure", allow_duplicate=True),
    Input("sim-method-select", "value"),
    State("model-store", "data"),
    State("tuning-store", "data"),
    State("selected-loop", "data"),
    State("loops-store", "data"),
    prevent_initial_call=True,
)
def update_simulation_with_method(method_name, model_data, tuning_data, selected, loops):
    if not model_data or not tuning_data or method_name is None:
        return dash.no_update

    K = model_data.get("K", 0)
    T = model_data.get("T", 0)
    L = model_data.get("L", 0)
    if K == 0 or T == 0:
        return dash.no_update

    controller_type = loops[selected]["controller_type"] if loops and selected >= 0 else "PID"
    action_dir = loops[selected]["action_direction"] if loops and selected >= 0 else "反作用"

    selected_params = None
    for key, params in tuning_data.items():
        if params.get("name") == method_name:
            selected_params = params
            break
    if selected_params is None:
        return dash.no_update

    Kp_new = selected_params["Kp"]
    Ki_new = selected_params["Ki"]
    Kd_new = selected_params["Kd"]

    sp_step = 10.0
    if loops and selected >= 0:
        pv_arr = np.array(loops[selected]["pv"])
        pv_range = np.max(pv_arr) - np.min(pv_arr)
        if pv_range > 0:
            sp_step = pv_range * 0.2

    t, sp, pv_new, pv_current = simulate_closed_loop(
        K, T, L, Kp_new, Ki_new, Kd_new,
        ts=1.0, sim_time=max(600, T * 10), sp_step_size=sp_step,
        controller_type=controller_type, action_direction=action_dir,
        current_Kp=Kp_new * 0.8, current_Ki=Ki_new * 0.5 if Ki_new else 0,
        current_Kd=Kd_new * 0.5 if Kd_new else 0,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sp, name="SP", line=dict(color=COLORS["sp"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=pv_new, name=f"新参数 ({method_name})",
                              line=dict(color=COLORS["green"], width=2)))
    if pv_current is not None:
        fig.add_trace(go.Scatter(x=t, y=pv_current, name="当前参数响应",
                                  line=dict(color=COLORS["pv"], width=1.5, dash="dot")))

    fig.update_layout(
        title="闭环阶跃响应仿真对比", paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
        font=dict(color="#ccc", size=10), height=300,
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(title="时间(s)", gridcolor="#333"),
        yaxis=dict(title="PV", gridcolor="#333"),
        legend=dict(orientation="h", y=1.12),
    )

    return fig


@app.callback(
    Output("selected-loop", "data", allow_duplicate=True),
    Input({"type": "rank-item", "index": ALL}, "n_clicks"),
    State("loops-store", "data"),
    prevent_initial_call=True,
)
def select_loop_from_ranking(clicks, loops):
    if not clicks or not loops:
        return dash.no_update

    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        idx_dict = json.loads(trigger_id)
        idx = idx_dict.get("index", -1)
        if 0 <= idx < len(loops):
            return idx
    except (json.JSONDecodeError, KeyError):
        pass

    return dash.no_update


@app.callback(
    Output("loop-list-container", "children", allow_duplicate=True),
    Output("health-ranking", "children", allow_duplicate=True),
    Input("area-filter-select", "value"),
    State("loops-store", "data"),
    State("selected-loop", "data"),
    prevent_initial_call=True,
)
def filter_by_area(selected_area, loops, selected_idx):
    if not loops:
        return [], []

    filtered_indices = []
    for i, loop in enumerate(loops):
        area = loop.get("area", "")
        if selected_area == "all" or area == selected_area or (selected_area == "" and not area):
            filtered_indices.append(i)

    loop_cards = []
    for i in filtered_indices:
        loop_cards.append(_loop_card(i, loops[i], i == selected_idx))

    ranking_items = _build_ranking(loops, selected_idx)

    return loop_cards, ranking_items


@app.callback(
    Output("area-filter-select", "options"),
    Input("loops-store", "data"),
)
def update_area_options(loops):
    if not loops:
        return [{"label": "全部区域", "value": "all"}]

    areas = set()
    for loop in loops:
        area = loop.get("area", "")
        if area:
            areas.add(area)

    options = [{"label": "全部区域", "value": "all"}]
    for area in sorted(areas):
        options.append({"label": area, "value": area})

    return options


@app.callback(
    Output("compare-modal", "is_open"),
    Output("compare-modal-body", "children"),
    Input("btn-compare", "n_clicks"),
    Input("btn-close-compare", "n_clicks"),
    State("loops-store", "data"),
    State("compare-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_compare_modal(open_clicks, close_clicks, loops, is_open):
    triggered = ctx.triggered_id
    if triggered == "btn-close-compare" or (triggered == "btn-compare" and is_open):
        return False, []

    if not loops or len(loops) < 2:
        return True, html.P("至少需要2个回路才能进行对比", className="text-muted")

    comparison_data = []
    metric_labels = {
        "IAE": "IAE",
        "ISE": "ISE",
        "ITAE": "ITAE",
        "overshoot_pct": "过冲量(%)",
        "settling_time": "调节时间(s)",
        "steady_state_error": "稳态误差",
        "oscillation_period": "振荡周期(s)",
        "decay_ratio": "衰减比",
    }

    all_metrics = {}
    for i, loop in enumerate(loops):
        pv = np.array(loop["pv"])
        sp = np.array(loop["sp"])
        co = np.array(loop["co"])
        ts = loop["sampling_period"]
        metrics = compute_metrics(pv, sp, co, ts, loop["controller_type"])
        all_metrics[loop["name"]] = metrics

    table_header = ["指标"] + [loop["name"] for loop in loops]
    table_rows = [table_header]

    for key, label in metric_labels.items():
        row = [label]
        for loop in loops:
            val = all_metrics[loop["name"]].get(key, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table_rows.append(row)

    comparison_table = html.Table(
        [html.Thead(html.Tr([html.Th(c, className="text-warning small p-2") for c in table_rows[0]]))] +
        [html.Tbody([html.Tr([
            html.Td(table_rows[i][0], className="text-light small p-2 fw-bold"),
        ] + [
            html.Td(table_rows[i][j], className="text-white small p-2")
            for j in range(1, len(table_rows[i]))
        ]) for i in range(1, len(table_rows))])],
        className="w-100 table table-dark table-striped",
        style={"fontSize": "12px", "borderCollapse": "collapse"}
    )

    harris_rows = [["回路", "Harris指标", "状态", "潜在改善(%)"]]
    for loop in loops:
        pv = np.array(loop["pv"])
        sp = np.array(loop["sp"])
        ts = loop["sampling_period"]
        h_val, improvement, _ = harris_index(pv, sp, ts, loop["controller_type"])
        status = "优秀" if h_val >= 0.8 else ("有提升空间" if h_val >= 0.5 else "急需调优")
        color = "success" if h_val >= 0.8 else ("warning" if h_val >= 0.5 else "danger")
        harris_rows.append([
            loop["name"],
            f"{h_val:.4f}",
            html.Span(status, className=f"text-{color} fw-bold"),
            f"{improvement:.1f}",
        ])

    harris_table = html.Table(
        [html.Thead(html.Tr([html.Th(c, className="text-warning small p-2") for c in harris_rows[0]]))] +
        [html.Tbody([html.Tr([
            html.Td(harris_rows[i][j], className="text-white small p-2")
            for j in range(len(harris_rows[i]))
        ]) for i in range(1, len(harris_rows))])],
        className="w-100 table table-dark table-striped",
        style={"fontSize": "12px", "borderCollapse": "collapse"}
    )

    content = html.Div([
        html.H6("性能指标对比", className="text-white mb-2"),
        comparison_table,
        html.Hr(className="my-4", style={"borderColor": "#333"}),
        html.H6("Harris指标对比", className="text-white mb-2"),
        harris_table,
    ])

    return True, content


def _compute_harris_segments(pv, sp, co, ts, controller_type):
    n = len(pv)
    if n < 40:
        return None

    seg_size = n // 4
    segments = []

    for i in range(4):
        start = i * seg_size
        end = start + seg_size if i < 3 else n
        pv_seg = pv[start:end]
        sp_seg = sp[start:end]
        co_seg = co[start:end]

        if len(pv_seg) < 10:
            segments.append({"harris": 0.5, "sp_changes": 0, "co_saturation_pct": 0, "pv_std_ratio": 1})
            continue

        h_val, _, _ = harris_index(pv_seg, sp_seg, ts, controller_type)

        sp_changes = np.sum(np.abs(np.diff(sp_seg)) > 0.5)

        co_max = np.max(co_seg)
        co_min = np.min(co_seg)
        co_range = co_max - co_min if co_max > co_min else 1
        at_max = np.sum(co_seg >= co_max - co_range * 0.02)
        at_min = np.sum(co_seg <= co_min + co_range * 0.02)
        co_saturation_pct = (at_max + at_min) / len(co_seg) * 100

        pv_std = np.std(pv_seg)
        overall_std = np.std(pv)
        pv_std_ratio = pv_std / overall_std if overall_std > 0 else 1

        segments.append({
            "harris": float(h_val),
            "sp_changes": int(sp_changes),
            "co_saturation_pct": float(co_saturation_pct),
            "pv_std_ratio": float(pv_std_ratio),
        })

    return segments


def _compute_spectrum(pv, ts):
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks

    n = len(pv)
    if n < 10:
        return {"freqs": None, "amps": None, "peaks": []}

    pv_detrend = pv - np.mean(pv)
    win = np.hanning(n)
    pv_win = pv_detrend * win

    yf = np.abs(fft(pv_win))[: n // 2]
    freqs = fftfreq(n, d=ts)[: n // 2]

    if len(yf) < 3:
        return {"freqs": None, "amps": None, "peaks": []}

    yf_norm = yf / np.max(yf) if np.max(yf) > 0 else yf

    peak_indices, peak_props = find_peaks(
        yf_norm,
        height=0.3,
        distance=max(1, int(n * ts * 0.01)),
    )

    peaks = []
    if len(peak_indices) > 0:
        peak_heights = yf[peak_indices]
        sorted_idx = np.argsort(peak_heights)[::-1]
        max_amp = np.max(yf) if np.max(yf) > 0 else 1

        for idx in sorted_idx[:5]:
            peak_idx = peak_indices[idx]
            freq = freqs[peak_idx]
            amp = yf[peak_idx]
            period = 1.0 / freq if freq > 0 else float('inf')

            meaning = _interpret_frequency(freq, period, ts, n)

            peaks.append({
                "freq": float(freq),
                "period": float(period),
                "amp": float(amp / max_amp),
                "meaning": meaning,
            })

    return {
        "freqs": freqs.tolist(),
        "amps": yf.tolist(),
        "peaks": peaks,
    }


def _interpret_frequency(freq, period, ts, n_points):
    if period < 5:
        return "高频成分，可能与阀门行程周期或传感器噪声有关"
    elif period < 30:
        return "中高频振荡，可能与控制器参数不当或阀门粘滞有关"
    elif period < 120:
        return "中频振荡，可能接近SP阶跃间隔或过程共振频率"
    elif period < 600:
        return "低频振荡，可能与上游扰动或工艺周期变化有关"
    else:
        return "超低频成分，可能与慢扰动或过程漂移有关"


@app.callback(
    Output("annotation-pending", "data"),
    Output("annotations-store", "data"),
    Output("annotation-status", "children"),
    Input("timeseries-graph", "clickData"),
    Input("btn-add-annotation", "n_clicks"),
    Input("btn-clear-annotations", "n_clicks"),
    Input({"type": "delete-annotation", "index": ALL}, "n_clicks"),
    State("annotation-pending", "data"),
    State("annotations-store", "data"),
    State("selected-loop", "data"),
    State("loops-store", "data"),
    State("time-range-store", "data"),
    prevent_initial_call=True,
)
def handle_annotations(click_data, add_clicks, clear_clicks, delete_clicks,
                        pending, annotations_store, selected_idx, loops, time_ranges):
    ctx = dash.callback_context
    triggered = ctx.triggered_id

    if not loops or selected_idx < 0:
        return dash.no_update, dash.no_update, ""

    annotations_store = dict(annotations_store) if annotations_store else {}
    loop_key = str(selected_idx)
    annotations = annotations_store.get(loop_key, [])
    status_msg = ""

    if triggered == "btn-clear-annotations":
        if clear_clicks is None or clear_clicks == 0:
            return dash.no_update, dash.no_update, ""
        annotations = []
        annotations_store[loop_key] = annotations
        return None, annotations_store, "已清除所有标注"

    if triggered == "btn-add-annotation":
        if add_clicks is None or add_clicks == 0:
            return dash.no_update, dash.no_update, ""
        if pending is None or pending.get("loop_idx") != selected_idx:
            return {"loop_idx": selected_idx, "step": 1}, dash.no_update, "请点击PV曲线上的起点"
        else:
            return None, dash.no_update, "已取消标注模式"

    if isinstance(triggered, dict) and triggered.get("type") == "delete-annotation":
        idx = triggered.get("index", 0)
        if 0 <= idx < len(annotations):
            del annotations[idx]
            annotations_store[loop_key] = annotations
        return dash.no_update, annotations_store, f"已删除标注{idx+1}"

    if triggered == "timeseries-graph" and click_data:
        points = click_data.get("points", [])
        if not points or len(points) == 0:
            if pending is not None and pending.get("loop_idx") == selected_idx:
                return None, dash.no_update, "已取消标注"
            return dash.no_update, dash.no_update, ""

        point = points[0]
        x_val = point.get("x", 0)
        y_val = point.get("y", 0)

        if pending is not None and pending.get("loop_idx") == selected_idx:
            step = pending.get("step", 1)
            if step == 1:
                new_pending = {
                    "loop_idx": selected_idx,
                    "step": 2,
                    "x": x_val,
                    "y": y_val,
                }
                return new_pending, dash.no_update, f"起点已选 ({x_val:.1f}s, {y_val:.2f})，请点击终点"
            elif step == 2:
                x0 = pending.get("x", 0)
                y0 = pending.get("y", 0)

                duration = abs(x_val - x0)
                if duration < 0.1:
                    return dash.no_update, dash.no_update, "两点距离太近，请重新选择终点"

                slope = (y_val - y0) / duration if duration > 0 else 0

                new_ann = {
                    "x0": min(x0, x_val),
                    "x1": max(x0, x_val),
                    "y0": y0,
                    "y1": y_val,
                    "slope": slope,
                    "duration": duration,
                }
                annotations.append(new_ann)
                annotations_store[loop_key] = annotations

                return None, annotations_store, f"标注{len(annotations)}已添加: 斜率={slope:.4f}/s, 时长={duration:.1f}s"

        return dash.no_update, dash.no_update, status_msg

    return dash.no_update, dash.no_update, status_msg


@app.callback(
    Output("annotation-tags", "children"),
    Input("annotations-store", "data"),
    State("selected-loop", "data"),
    prevent_initial_call=True,
)
def update_annotation_tags(annotations_store, selected_idx):
    if not annotations_store or selected_idx < 0:
        return []

    loop_key = str(selected_idx)
    annotations = annotations_store.get(loop_key, [])

    buttons = []
    for i, ann in enumerate(annotations):
        buttons.append(
            dbc.Button(
                [html.I(className="fas fa-times me-1"), f"标注{i+1}"],
                id={"type": "delete-annotation", "index": i},
                size="sm",
                color="warning",
                outline=True,
                className="me-1 mb-1",
                style={"fontSize": "11px"},
            )
        )

    return buttons


@app.callback(
    Output("metrics-history-store", "data"),
    Output("record-status", "children"),
    Input("btn-record-metrics", "n_clicks"),
    Input("btn-clear-history-modal", "n_clicks"),
    State("metrics-history-store", "data"),
    State("analysis-store", "data"),
    State("selected-loop", "data"),
    State("time-range-store", "data"),
    prevent_initial_call=True,
)
def handle_metrics_history(record_clicks, clear_clicks, history_store, analysis,
                            selected_idx, time_ranges):
    ctx = dash.callback_context
    triggered = ctx.triggered_id

    if selected_idx < 0:
        return dash.no_update, ""

    history_store = dict(history_store) if history_store else {}
    loop_key = str(selected_idx)
    history = history_store.get(loop_key, [])

    if triggered == "btn-clear-history-modal":
        if clear_clicks is None or clear_clicks == 0:
            return dash.no_update, ""
        history_store[loop_key] = []
        return history_store, "历史记录已清空"

    if triggered == "btn-record-metrics":
        if record_clicks is None or record_clicks == 0:
            return dash.no_update, ""
        if not analysis or "metrics" not in analysis:
            return dash.no_update, "暂无指标数据可记录"

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        time_range = time_ranges.get(loop_key, {}) if time_ranges else {}
        if time_range and "x" in time_range:
            range_desc = f"{time_range['x'][0]:.1f}s - {time_range['x'][1]:.1f}s"
        else:
            range_desc = "全部数据"

        record = {
            "timestamp": timestamp,
            "metrics": analysis["metrics"],
            "harris": analysis.get("harris", {}),
            "range": range_desc,
        }

        history.append(record)

        if len(history) > 10:
            history = history[-10:]

        history_store[loop_key] = history

        return history_store, f"已记录快照 #{len(history)} ({timestamp})"

    return dash.no_update, ""


def _build_history_content(history_store, selected_idx):
    if selected_idx < 0 or not history_store:
        return html.P("暂无历史记录，请先点击\"记录当前指标\"保存快照", className="text-muted")

    loop_key = str(selected_idx)
    history = history_store.get(loop_key, [])

    if not history:
        return html.P("暂无历史记录，请先点击\"记录当前指标\"保存快照", className="text-muted")

    metric_names = ["IAE", "ISE", "ITAE", "overshoot_pct", "settling_time",
                    "steady_state_error", "oscillation_period", "decay_ratio"]
    metric_labels = {
        "IAE": "IAE",
        "ISE": "ISE",
        "ITAE": "ITAE",
        "overshoot_pct": "过冲量(%)",
        "settling_time": "调节时间(s)",
        "steady_state_error": "稳态误差",
        "oscillation_period": "振荡周期(s)",
        "decay_ratio": "衰减比",
    }

    timestamps = [h["timestamp"] for h in history]

    figs = []
    for metric in metric_names:
        values = [h["metrics"].get(metric, 0) for h in history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode="lines+markers",
            name=metric_labels[metric],
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6),
        ))
        fig.update_layout(
            title=dict(text=metric_labels[metric], font=dict(color="white", size=12)),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0a1628",
            font=dict(color="#ccc", size=10),
            height=200,
            margin=dict(l=50, r=20, t=30, b=40),
            xaxis=dict(gridcolor="#333", tickangle=30, tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#333"),
            showlegend=False,
        )
        figs.append(dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}),
                            width=4, className="mb-3"))

    table_rows = []
    table_rows.append(html.Tr(
        [html.Th("时间", className="text-warning small p-2")] +
        [html.Th(metric_labels[m], className="text-warning small p-2 text-center") for m in metric_names]
    ))

    for i, h in enumerate(history):
        row = [html.Td(h["timestamp"], className="text-light small p-2")]
        for m in metric_names:
            val = h["metrics"].get(m, "N/A")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)

            if i > 0:
                prev_val = history[i-1]["metrics"].get(m, 0)
                if prev_val and prev_val != 0 and not (isinstance(prev_val, float) and np.isnan(prev_val)):
                    change = (val - prev_val) / abs(prev_val) * 100
                    if abs(change) > 10:
                        color = COLORS["red"]
                        arrow = "↑" if change > 0 else "↓"
                        val_str = f"{val_str} {arrow}"
                        row.append(html.Td(val_str, className="text-white small p-2 text-center",
                                           style={"color": color, "fontWeight": "bold"}))
                    else:
                        row.append(html.Td(val_str, className="text-white small p-2 text-center"))
                else:
                    row.append(html.Td(val_str, className="text-white small p-2 text-center"))
            else:
                row.append(html.Td(val_str, className="text-white small p-2 text-center"))
        table_rows.append(html.Tr(row))

    history_table = html.Table(
        table_rows,
        className="w-100 table table-dark table-striped",
        style={"fontSize": "11px", "borderCollapse": "collapse"}
    )

    content = html.Div([
        html.H6("指标变化趋势", className="text-white mb-3"),
        dbc.Row(figs),
        html.Hr(className="my-4", style={"borderColor": "#333"}),
        html.H6("历史记录详情", className="text-white mb-2"),
        html.Div(className="table-responsive", children=[history_table]),
    ])

    return content


@app.callback(
    Output("history-modal", "is_open"),
    Output("history-modal-body", "children"),
    Input("btn-view-history", "n_clicks"),
    Input("btn-close-history", "n_clicks"),
    State("metrics-history-store", "data"),
    State("selected-loop", "data"),
    State("history-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_history_modal(view_clicks, close_clicks, history_store, selected_idx, is_open):
    ctx = dash.callback_context
    triggered = ctx.triggered_id

    if triggered == "btn-close-history":
        if close_clicks is None or close_clicks == 0:
            return dash.no_update, dash.no_update
        return False, []

    if triggered == "btn-view-history":
        if view_clicks is None or view_clicks == 0:
            return dash.no_update, dash.no_update
        if is_open:
            return False, []
        else:
            content = _build_history_content(history_store, selected_idx)
            return True, content

    return dash.no_update, dash.no_update


@app.callback(
    Output("history-modal-body", "children", allow_duplicate=True),
    Input("metrics-history-store", "data"),
    State("selected-loop", "data"),
    State("history-modal", "is_open"),
    prevent_initial_call=True,
)
def update_history_modal_content(history_store, selected_idx, is_open):
    if not is_open:
        return dash.no_update
    return _build_history_content(history_store, selected_idx)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
