import numpy as np
from typing import Dict, List, Optional, Tuple


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    对序列做线性回归拟合
    返回: (斜率slope, 截距intercept, R²决定系数)
    """
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n > 0 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-12:
        return 0.0, y_mean, 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if ss_tot < 1e-12:
        r_squared = 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    return float(slope), float(intercept), float(r_squared)


def extract_loop_sequences(inspection_history: List[Dict], loop_idx: int,
                            max_rounds: int = 30) -> Dict[str, List[float]]:
    """
    从inspection-history-store中提取指定回路的历史指标序列
    返回包含各指标序列的字典
    """
    if not inspection_history:
        return {
            "health_score": [],
            "harris_index": [],
            "osc_amplitude_pct": [],
            "stiction_score": [],
        }

    health_scores = []
    harris_indices = []
    osc_amplitudes = []
    stiction_scores = []

    recent_rounds = inspection_history[-max_rounds:] if len(inspection_history) > max_rounds else inspection_history

    for rnd in recent_rounds:
        for lr in rnd.get("loop_results", []):
            if lr.get("loop_idx") == loop_idx:
                health_scores.append(lr.get("health_score", 0.0))
                harris_indices.append(lr.get("harris_index", 0.0) * 100)
                osc_amplitudes.append(lr.get("osc_amplitude_pct", 0.0))

                stiction_severity = lr.get("stiction_severity", "无")
                severity_map = {"无": 100.0, "轻微": 70.0, "中度": 40.0, "严重": 10.0}
                stiction_scores.append(severity_map.get(stiction_severity, 100.0))
                break

    return {
        "health_score": health_scores,
        "harris_index": harris_indices,
        "osc_amplitude_pct": osc_amplitudes,
        "stiction_score": stiction_scores,
    }


def build_degradation_model(score_sequence: List[float]) -> Dict:
    """
    对单条指标序列建立劣化模型
    返回: {slope, intercept, r_squared, is_significant, current_value}
    """
    if len(score_sequence) < 3:
        return {
            "slope": 0.0,
            "intercept": float(score_sequence[-1]) if score_sequence else 0.0,
            "r_squared": 0.0,
            "is_significant": False,
            "current_value": float(score_sequence[-1]) if score_sequence else 0.0,
            "n_points": len(score_sequence),
        }

    x = np.arange(len(score_sequence), dtype=float)
    y = np.array(score_sequence, dtype=float)

    slope, intercept, r_squared = linear_regression(x, y)

    is_significant = r_squared > 0.6 and slope < -0.5

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "is_significant": is_significant,
        "current_value": float(score_sequence[-1]) if len(score_sequence) > 0 else 0.0,
        "n_points": len(score_sequence),
    }


def predict_failure_round(model: Dict, threshold: float) -> Optional[float]:
    """
    基于线性模型预测指标跌破阈值的剩余轮次
    返回剩余轮次，如果不会跌破则返回None
    """
    slope = model["slope"]
    intercept = model["intercept"]
    current_round = model["n_points"] - 1

    if slope >= 0:
        return None

    current_value = slope * current_round + intercept

    if current_value <= threshold:
        return 0.0

    rounds_to_threshold = (threshold - intercept) / slope - current_round

    if rounds_to_threshold > 0:
        return float(rounds_to_threshold)

    return None


def analyze_loop_degradation(inspection_history: List[Dict], loop_idx: int,
                              warning_threshold: float = 60.0,
                              danger_threshold: float = 40.0) -> Dict:
    """
    分析单个回路的完整劣化情况
    """
    sequences = extract_loop_sequences(inspection_history, loop_idx)

    health_model = build_degradation_model(sequences["health_score"])
    harris_model = build_degradation_model(sequences["harris_index"])
    osc_model = build_degradation_model(sequences["osc_amplitude_pct"])

    osc_degradation_slope = -osc_model["slope"]

    stiction_model = build_degradation_model(sequences["stiction_score"])

    warning_rounds = predict_failure_round(health_model, warning_threshold)
    danger_rounds = predict_failure_round(health_model, danger_threshold)

    sub_metrics = {
        "harris_index": {
            "model": harris_model,
            "degradation_rate": -harris_model["slope"],
            "label": "Harris指标",
        },
        "oscillation": {
            "model": osc_model,
            "degradation_rate": osc_degradation_slope,
            "label": "振荡幅度",
        },
        "stiction": {
            "model": stiction_model,
            "degradation_rate": -stiction_model["slope"],
            "label": "粘滞程度",
        },
        "health_score": {
            "model": health_model,
            "degradation_rate": -health_model["slope"],
            "label": "综合健康评分",
        },
    }

    fastest_degrading = max(
        sub_metrics.items(),
        key=lambda x: x[1]["degradation_rate"]
    )

    return {
        "loop_idx": loop_idx,
        "health_model": health_model,
        "harris_model": harris_model,
        "osc_model": osc_model,
        "stiction_model": stiction_model,
        "warning_rounds": warning_rounds,
        "danger_rounds": danger_rounds,
        "warning_threshold": warning_threshold,
        "danger_threshold": danger_threshold,
        "sub_metrics": sub_metrics,
        "fastest_degrading_key": fastest_degrading[0],
        "fastest_degrading_label": fastest_degrading[1]["label"],
        "fastest_degrading_rate": fastest_degrading[1]["degradation_rate"],
        "sequences": sequences,
        "is_significant": health_model["is_significant"],
    }


def analyze_all_loops(inspection_history: List[Dict], loops: List[Dict],
                       warning_threshold: float = 60.0,
                       danger_threshold: float = 40.0) -> List[Dict]:
    """
    分析所有回路的劣化情况，按严重程度排序
    """
    if not loops or not inspection_history:
        return []

    results = []
    for i, loop in enumerate(loops):
        analysis = analyze_loop_degradation(
            inspection_history, i, warning_threshold, danger_threshold
        )
        analysis["loop_name"] = loop.get("name", f"回路{i}")
        analysis["loop_idx"] = i
        results.append(analysis)

    def sort_key(x):
        warning = x["warning_rounds"]
        health_model = x.get("health_model", {})
        current_score = health_model.get("current_value", 100)
        slope = health_model.get("slope", 0)

        if warning is not None:
            return (0, warning, current_score, slope)
        elif x["is_significant"]:
            return (1, -slope, current_score)
        else:
            return (2, -current_score, slope)

    results.sort(key=sort_key)
    return results


def generate_maintenance_suggestions(degradation_analysis: Dict,
                                      inspection_interval_seconds: float = 60.0) -> List[Dict]:
    """
    根据劣化分析结果生成维护建议
    返回建议列表，按优先级排序
    """
    suggestions = []

    warning_rounds = degradation_analysis.get("warning_rounds")
    danger_rounds = degradation_analysis.get("danger_rounds")
    fastest_key = degradation_analysis.get("fastest_degrading_key")
    fastest_label = degradation_analysis.get("fastest_degrading_label")

    if warning_rounds is not None and warning_rounds <= 3:
        priority = "紧急"
    elif warning_rounds is not None and warning_rounds <= 10:
        priority = "高"
    elif degradation_analysis.get("is_significant", False):
        priority = "中"
    else:
        priority = "低"

    danger_hours = None
    if danger_rounds is not None:
        danger_hours = danger_rounds * inspection_interval_seconds / 3600.0

    if fastest_key == "harris_index":
        suggestions.append({
            "priority": priority,
            "action": "检查阀门行程与执行机构性能",
            "description": f"Harris指标劣化最快(速率:{degradation_analysis['fastest_degrading_rate']:.2f}/轮)，建议检查阀门定位器、执行机构死区及摩擦特性",
            "maintenance_window_hours": danger_hours,
            "type": "valve",
        })
        suggestions.append({
            "priority": _lower_priority(priority),
            "action": "重新整定PID参数",
            "description": "控制性能下降，建议基于当前过程特性重新整定PID参数",
            "maintenance_window_hours": danger_hours,
            "type": "tuning",
        })
    elif fastest_key == "oscillation":
        suggestions.append({
            "priority": priority,
            "action": "检查振荡源并排查整定参数",
            "description": f"振荡幅度加剧最快(速率:{degradation_analysis['fastest_degrading_rate']:.2f}/轮)，可能是PID参数过紧或阀门振荡",
            "maintenance_window_hours": danger_hours,
            "type": "oscillation",
        })
        suggestions.append({
            "priority": _lower_priority(priority),
            "action": "适当降低控制器增益",
            "description": "振荡加剧时可适当减小比例增益或增加积分时间以抑制振荡",
            "maintenance_window_hours": danger_hours,
            "type": "tuning",
        })
    elif fastest_key == "stiction":
        suggestions.append({
            "priority": priority,
            "action": "检查阀门粘滞与润滑情况",
            "description": f"粘滞程度劣化最快(速率:{degradation_analysis['fastest_degrading_rate']:.2f}/轮)，建议检查阀门填料、润滑及气源压力",
            "maintenance_window_hours": danger_hours,
            "type": "valve",
        })
        suggestions.append({
            "priority": _lower_priority(priority),
            "action": "考虑阀门维护或更换",
            "description": "粘滞严重时可能需要阀门解体检修或更换",
            "maintenance_window_hours": danger_hours,
            "type": "valve",
        })
    else:
        suggestions.append({
            "priority": priority,
            "action": "综合性能评估与全面检查",
            "description": "综合健康评分呈下降趋势，建议对回路进行全面性能评估",
            "maintenance_window_hours": danger_hours,
            "type": "general",
        })

    if degradation_analysis.get("is_significant", False):
        suggestions.append({
            "priority": _lower_priority(priority),
            "action": "检查传感器漂移与校准",
            "description": "性能持续劣化可能与传感器漂移有关，建议检查并校准测量仪表",
            "maintenance_window_hours": danger_hours,
            "type": "sensor",
        })

    suggestions = suggestions[:3]

    priority_order = {"紧急": 0, "高": 1, "中": 2, "低": 3}
    suggestions.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return suggestions


def _lower_priority(priority: str) -> str:
    priority_map = {"紧急": "高", "高": "中", "中": "低", "低": "低"}
    return priority_map.get(priority, "低")


def get_degradation_arrow(slope: float, r_squared: float) -> str:
    """
    根据劣化速率返回方向箭头
    """
    if r_squared < 0.3:
        return "→"
    if slope > 0.3:
        return "↗"
    elif slope < -0.5:
        return "↓"
    elif slope < -0.2:
        return "↘"
    else:
        return "→"
