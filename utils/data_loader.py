import io
import base64
import pandas as pd
import numpy as np


def parse_csv(contents, filename, sampling_period, controller_type, action_direction, area=""):
    content_type, content_string = contents.split(",")
    decoded = io.BytesIO(base64.b64decode(content_string))
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(decoded)
        else:
            df = pd.read_excel(decoded)
    except Exception:
        return None, f"无法解析文件 {filename}"

    required_cols = 3
    if len(df.columns) < required_cols:
        return None, f"数据列不足，至少需要时间戳、PV、CO三列"

    ts_col = df.columns[0]
    ts = pd.to_datetime(df[ts_col], errors="coerce")

    pv_col = None
    sp_col = None
    co_col = None

    for col in df.columns[1:]:
        col_lower = col.strip().lower()
        if "pv" in col_lower or "process" in col_lower:
            pv_col = col
        elif "sp" in col_lower or "setpoint" in col_lower or "set" in col_lower:
            sp_col = col
        elif "co" in col_lower or "output" in col_lower or "mv" in col_lower:
            co_col = col

    if pv_col is None:
        pv_col = df.columns[1]
    if co_col is None:
        for col in df.columns[1:]:
            if col != pv_col and col != sp_col:
                co_col = col
                break
    if co_col is None:
        co_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]

    if sp_col is None:
        for col in df.columns[1:]:
            if col != pv_col and col != co_col:
                sp_col = col
                break
    if sp_col is None:
        sp_col = pv_col

    pv = pd.to_numeric(df[pv_col], errors="coerce").values
    co = pd.to_numeric(df[co_col], errors="coerce").values
    sp = pd.to_numeric(df[sp_col], errors="coerce").values if sp_col != pv_col else pv.copy()

    mask = ~(np.isnan(pv) | np.isnan(co) | np.isnan(sp))
    pv = pv[mask]
    co = co[mask]
    sp = sp[mask]
    ts = ts[mask]

    if len(pv) < 10:
        return None, "有效数据点不足（少于10个）"

    loop_data = {
        "name": filename.replace(".csv", "").replace(".xlsx", ""),
        "timestamp": ts,
        "pv": pv,
        "sp": sp,
        "co": co,
        "sampling_period": float(sampling_period),
        "controller_type": controller_type,
        "action_direction": action_direction,
        "area": area,
        "n_points": len(pv),
    }

    return loop_data, None


def create_demo_loop(loop_name="Demo-TIC-101", area="反应区", n_points=2000, ts=1.0):
    t = np.arange(n_points) * ts

    K = 0.4
    tau = 50.0
    L = 10.0

    Kp = 1.5
    Ti = 60.0
    Ki = Kp / Ti
    Kd = 0.0

    sp = np.ones(n_points) * 50.0
    sp[300:] = 60.0
    sp[900:] = 55.0
    sp[1500:] = 65.0

    pv = np.zeros(n_points)
    co = np.zeros(n_points)
    pv[0] = 50.0
    co0 = 50.0 / K / 10.0
    integral = 0.0
    prev_err = 0.0

    for i in range(1, n_points):
        err = sp[i] - pv[i - 1]
        integral += err * ts
        derivative = (err - prev_err) / ts
        co_val = co0 + Kp * err + Ki * integral + Kd * derivative
        co_val = np.clip(co_val, 0, 100)
        if co_val <= 0 or co_val >= 100:
            integral -= err * ts
        co[i] = co_val

        delay_samples = int(L / ts)
        delay_idx = max(0, i - delay_samples)
        u = co[delay_idx]
        dpv = (K * u * 10.0 - pv[i - 1]) / tau * ts
        noise = np.random.normal(0, 0.1)
        pv[i] = pv[i - 1] + dpv + noise
        prev_err = err

    timestamps = pd.date_range("2024-01-01", periods=n_points, freq=f"{ts}s")

    loop_data = {
        "name": loop_name,
        "timestamp": timestamps,
        "pv": pv,
        "sp": sp,
        "co": co,
        "sampling_period": ts,
        "controller_type": "PI",
        "action_direction": "反作用",
        "area": area,
        "n_points": n_points,
    }
    return loop_data

