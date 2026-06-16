import numpy as np


def detect_stiction(pv, sp, co, ts):
    err = pv - sp
    results = {
        "has_stiction": False,
        "stiction_severity": "无",
        "dead_band_width": 0.0,
        "reversal_count": 0,
        "reversal_rate": 0.0,
        "recommendation": "",
        "ellipse_score": 0.0,
    }

    co_c = co - np.mean(co)
    pv_c = pv - np.mean(pv)

    ellipse_score = _compute_ellipse_score(co_c, pv_c)
    results["ellipse_score"] = float(ellipse_score)

    reversal_count = _count_reversals(co)
    results["reversal_count"] = reversal_count
    duration = len(co) * ts
    results["reversal_rate"] = reversal_count / duration * 60 if duration > 0 else 0

    dead_band = _estimate_dead_band(co, pv)
    results["dead_band_width"] = float(dead_band)

    if ellipse_score > 0.4 or (reversal_count > len(co) * 0.15):
        results["has_stiction"] = True

        if ellipse_score > 0.7 or reversal_count > len(co) * 0.3:
            results["stiction_severity"] = "严重"
            results["recommendation"] = "立即处理 - 阀门粘滞严重影响控制品质，建议立即检修或更换阀门。"
        elif ellipse_score > 0.55 or reversal_count > len(co) * 0.2:
            results["stiction_severity"] = "中度"
            results["recommendation"] = "计划检修 - 阀门存在中等程度粘滞，建议安排近期检修。"
        else:
            results["stiction_severity"] = "轻微"
            results["recommendation"] = "持续观察 - 阀门轻微粘滞，暂不影响控制品质，建议持续监控。"

    return results


def _compute_ellipse_score(co_c, pv_c):
    n = min(len(co_c), len(pv_c))
    if n < 20:
        return 0.0

    co_norm = (co_c - np.min(co_c)) / (np.max(co_c) - np.min(co_c) + 1e-10)
    pv_norm = (pv_c - np.min(pv_c)) / (np.max(pv_c) - np.min(pv_c) + 1e-10)

    directions = np.zeros(n)
    for i in range(1, n):
        if co_norm[i] > co_norm[i - 1] + 1e-6:
            directions[i] = 1
        elif co_norm[i] < co_norm[i - 1] - 1e-6:
            directions[i] = -1

    up_idx = directions > 0
    down_idx = directions < 0

    if np.sum(up_idx) < 5 or np.sum(down_idx) < 5:
        return 0.0

    co_up = co_norm[up_idx]
    pv_up = pv_norm[up_idx]
    co_down = co_norm[down_idx]
    pv_down = pv_norm[down_idx]

    try:
        coeff_up = np.polyfit(co_up, pv_up, 1)
        coeff_down = np.polyfit(co_down, pv_down, 1)
    except Exception:
        return 0.0

    co_range = np.linspace(0, 1, 50)
    pv_up_fit = np.polyval(coeff_up, co_range)
    pv_down_fit = np.polyval(coeff_down, co_range)

    area = np.mean(np.abs(pv_up_fit - pv_down_fit))

    pv_span = np.max(pv_norm) - np.min(pv_norm) + 1e-10
    score = area / pv_span
    score = np.clip(score, 0, 1)

    return float(score)


def _count_reversals(co, min_change=0.1):
    reversals = 0
    prev_dir = 0
    for i in range(1, len(co)):
        diff = co[i] - co[i - 1]
        if abs(diff) > min_change:
            current_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and current_dir != prev_dir:
                reversals += 1
            prev_dir = current_dir
    return reversals


def _estimate_dead_band(co, pv):
    n = min(len(co), len(pv))
    if n < 20:
        return 0.0

    co_sorted_idx = np.argsort(co)
    co_sorted = co[co_sorted_idx]
    pv_sorted = pv[co_sorted_idx]

    window = max(5, n // 50)
    pv_spread = []
    for i in range(0, n - window, window):
        segment = pv_sorted[i:i + window]
        pv_spread.append(np.max(segment) - np.min(segment))

    if not pv_spread:
        return 0.0

    avg_spread = np.mean(pv_spread)
    co_range = np.max(co) - np.min(co)
    pv_range = np.max(pv) - np.min(pv)

    if pv_range < 1e-10:
        return 0.0

    dead_band = avg_spread / pv_range * co_range

    return float(dead_band)
