import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks


def identify_fopdt(pv, sp, co, ts, method="area", step_start=None, step_end=None):
    if step_start is not None and step_end is not None:
        i_start = step_start
        i_end = step_end
    else:
        i_start, i_end = _auto_detect_step(sp, ts)

    if i_start is None or i_end is None:
        return None, "未检测到阶跃变化，请手动标记或使用继电反馈法"

    if i_end - i_start < 10:
        return None, "阶跃段数据点不足"

    pv_seg = pv[i_start:i_end]
    co_seg = co[i_start:i_end]
    sp_seg = sp[i_start:i_end]
    t_seg = np.arange(len(pv_seg)) * ts

    co_before = co[max(i_start - 5, 0):i_start]
    if len(co_before) == 0:
        co_before = np.array([co_seg[0]])
    co_init = np.mean(co_before)

    pv_before = pv[max(i_start - 5, 0):i_start]
    if len(pv_before) == 0:
        pv_before = np.array([pv_seg[0]])
    pv_init = np.mean(pv_before)

    co_step = np.mean(co_seg[len(co_seg) // 4:])
    delta_co = co_step - co_init

    if abs(delta_co) < 0.05:
        return None, "CO阶跃幅度太小，无法辨识模型"

    pv_final = np.mean(pv_seg[-len(pv_seg) // 4:])
    delta_pv = pv_final - pv_init

    K = delta_pv / delta_co if abs(delta_co) > 1e-6 else 0

    if method == "area":
        params = _area_method(pv_seg, pv_init, K, delta_co, ts)
    else:
        params = _optimization_method(pv_seg, co_seg, pv_init, co_init, ts, K)

    if params is None:
        return None, "模型辨识失败"

    K_id, T_id, L_id = params
    model_response = _simulate_fopdt(K_id, T_id, L_id, co_seg, co_init, pv_init, ts)

    residuals = pv_seg - model_response
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((pv_seg - np.mean(pv_seg)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r_squared = max(0.0, min(1.0, r_squared))

    result = {
        "K": float(K_id),
        "T": float(T_id),
        "L": float(L_id),
        "r_squared": float(r_squared),
        "method": method,
        "model_response": model_response,
        "t_seg": t_seg,
        "pv_seg": pv_seg,
        "step_range": (i_start, i_end),
    }

    return result, None


def relay_feedback_identify(pv, sp, co, ts):
    err = pv - sp
    err_c = err - np.mean(err)

    zero_crossings = []
    for i in range(1, len(err_c)):
        if err_c[i - 1] * err_c[i] < 0:
            zero_crossings.append(i)

    if len(zero_crossings) < 4:
        return None, "数据中未检测到足够振荡周期，无法使用继电反馈法"

    periods = []
    for i in range(2, len(zero_crossings)):
        periods.append((zero_crossings[i] - zero_crossings[i - 2]) * ts)

    Pu = np.mean(periods[:min(5, len(periods))])

    amplitude = np.mean(np.abs(err_c))
    d = np.max(co) - np.min(co)
    if d < 1e-6:
        d = 1.0

    Ku = 4 * d / (np.pi * amplitude) if amplitude > 1e-6 else 0

    result = {
        "Ku": float(Ku),
        "Pu": float(Pu),
        "method": "relay_feedback",
    }

    return result, None


def _auto_detect_step(sp, ts):
    sp_diff = np.abs(np.diff(sp))
    if len(sp_diff) == 0:
        return None, None

    threshold = np.std(sp_diff) * 3 + np.mean(sp_diff)
    step_indices = np.where(sp_diff > max(threshold, 0.5))[0]

    if len(step_indices) == 0:
        return None, None

    i_start = step_indices[0]
    end_search = min(i_start + int(500 / ts), len(sp))
    i_end = end_search

    return i_start, i_end


def _area_method(pv_seg, pv_init, K, delta_co, ts):
    n = len(pv_seg)
    pv_response = pv_seg - pv_init

    total_area = np.sum(pv_response) * ts

    pv_inf = K * delta_co
    if abs(pv_inf) < 1e-10:
        pv_inf = np.mean(pv_response[-max(50, n//4):])
        if abs(pv_inf) < 1e-10:
            return None

    t63 = 0
    target = abs(pv_inf) * 0.632
    for i in range(n):
        if abs(pv_response[i]) >= target:
            t63 = i * ts
            break

    if t63 <= 0:
        t63 = n * ts * 0.4

    if pv_inf != 0:
        L = max(t63 - total_area / pv_inf, 0)
    else:
        L = t63 * 0.3
    T = t63 - L

    if T < ts * 3:
        T = ts * 5
    if L < ts:
        L = ts
    if L > t63 * 0.8:
        L = t63 * 0.25
        T = t63 - L

    return K, T, L


def _optimization_method(pv_seg, co_seg, pv_init, co_init, ts, K_init):
    n = len(pv_seg)

    def objective(params):
        K, T, L = params
        if T <= 0 or K == 0:
            return 1e10
        model = _simulate_fopdt(K, T, max(L, 0), co_seg, co_init, pv_init, ts)
        return np.sum((pv_seg - model) ** 2)

    x0 = [K_init, 30.0, 5.0]
    bounds = [(K_init * 0.1, K_init * 10) if K_init != 0 else (-10, 10),
              (1.0, 500.0),
              (0.0, 100.0)]

    try:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        K, T, L = result.x
        return K, T, L
    except Exception:
        return None


def _simulate_fopdt(K, T, L, co_seg, co_init, pv_init, ts):
    n = len(co_seg)
    pv_model = np.zeros(n)
    pv_model[0] = pv_init

    delay_samples = max(0, int(L / ts))

    for i in range(1, n):
        delayed_idx = max(0, i - delay_samples)
        if delayed_idx < len(co_seg):
            u = co_seg[delayed_idx]
        else:
            u = co_init

        dpv = (K * (u - co_init) - (pv_model[i - 1] - pv_init)) / T * ts
        pv_model[i] = pv_model[i - 1] + dpv

    return pv_model
