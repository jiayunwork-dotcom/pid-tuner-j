import numpy as np
from scipy import signal


def compute_metrics(pv, sp, co, ts, controller_type="PID"):
    err = sp - pv
    results = {}

    results["IAE"] = float(np.sum(np.abs(err)) * ts)
    results["ISE"] = float(np.sum(err ** 2) * ts)
    results["ITAE"] = float(np.sum(np.arange(len(err)) * ts * np.abs(err)) * ts)

    if np.max(np.abs(sp)) > 0:
        sp_range = np.max(sp) - np.min(sp) if np.max(sp) != np.min(sp) else np.max(np.abs(sp))
    else:
        sp_range = 1.0

    sp_changes = _detect_setpoint_changes(sp)
    if sp_changes:
        overshoots = []
        settling_times = []
        for idx in sp_changes:
            o, st = _compute_step_response_metrics(pv, sp, idx, ts, sp_range)
            if o is not None:
                overshoots.append(o)
            if st is not None:
                settling_times.append(st)
        results["overshoot_pct"] = float(np.mean(overshoots)) if overshoots else 0.0
        results["settling_time"] = float(np.mean(settling_times)) if settling_times else 0.0
    else:
        results["overshoot_pct"] = 0.0
        results["settling_time"] = 0.0

    tail_len = min(len(err), max(int(len(err) * 0.2), 50))
    results["steady_state_error"] = float(np.mean(err[-tail_len:]))

    osc_period = _detect_oscillation_period(pv, ts)
    results["oscillation_period"] = float(osc_period) if osc_period else 0.0

    decay_ratio = _compute_decay_ratio(pv, sp, ts)
    results["decay_ratio"] = float(decay_ratio) if decay_ratio else 0.0

    return results


def _detect_setpoint_changes(sp, threshold=0.5):
    changes = []
    for i in range(1, len(sp)):
        if abs(sp[i] - sp[i - 1]) > threshold:
            changes.append(i)
    merged = []
    for c in changes:
        if not merged or c - merged[-1] > 5:
            merged.append(c)
    return merged


def _compute_step_response_metrics(pv, sp, sp_idx, ts, sp_range):
    end_idx = min(sp_idx + int(300 / ts), len(pv))
    if end_idx - sp_idx < 10:
        return None, None

    segment_sp = sp[sp_idx:end_idx]
    segment_pv = pv[sp_idx:end_idx]

    new_sp = segment_sp[-1]
    old_sp = sp[max(sp_idx - 1, 0)]
    step_size = new_sp - old_sp

    if abs(step_size) < 1e-6:
        return None, None

    if step_size > 0:
        peak_val = np.max(segment_pv)
        overshoot = (peak_val - new_sp) / abs(step_size) * 100 if abs(step_size) > 0 else 0
    else:
        peak_val = np.min(segment_pv)
        overshoot = (new_sp - peak_val) / abs(step_size) * 100 if abs(step_size) > 0 else 0

    overshoot = max(0.0, overshoot)

    band = abs(step_size) * 0.02
    settled = None
    for i in range(len(segment_pv)):
        if i > 5 and abs(segment_pv[i] - new_sp) <= band:
            j = i
            while j < len(segment_pv) and abs(segment_pv[j] - new_sp) <= band:
                j += 1
            if j == len(segment_pv):
                settled = i
                break
            if j - i >= 10:
                settled = i
                break

    settling_time = settled * ts if settled is not None else None

    return overshoot, settling_time


def _detect_oscillation_period(pv, ts):
    from scipy.fft import fft

    n = len(pv)
    pv_detrend = pv - np.mean(pv)

    win = np.hanning(n)
    pv_win = pv_detrend * win

    yf = np.abs(fft(pv_win))[: n // 2]
    freqs = np.fft.fftfreq(n, d=ts)[: n // 2]

    if len(yf) < 3:
        return None

    yf[0] = 0
    peak_idx = np.argmax(yf)
    if yf[peak_idx] < np.std(pv_detrend) * 2:
        return None

    if freqs[peak_idx] > 0:
        period = 1.0 / freqs[peak_idx]

        acf_vals = _compute_acf(pv_detrend, nlags=min(int(period * 3 / ts), len(pv) // 2))
        if acf_vals is not None and len(acf_vals) > 2:
            if np.any(acf_vals[1:] < 0):
                return period

    return None


def _compute_decay_ratio(pv, sp, ts):
    pv_centered = pv - sp
    peaks, _ = signal.find_peaks(pv_centered, distance=max(3, int(5 / ts)))
    valleys, _ = signal.find_peaks(-pv_centered, distance=max(3, int(5 / ts)))

    all_extrema = np.sort(np.concatenate([peaks, valleys]))
    if len(all_extrema) < 4:
        return None

    extrem_vals = np.abs(pv_centered[all_extrema])
    if extrem_vals[0] < 1e-6:
        return None

    pairs = []
    for i in range(len(extrem_vals) - 1):
        if extrem_vals[i] > 1e-6:
            pairs.append(extrem_vals[i + 1] / extrem_vals[i])

    if pairs:
        return np.mean(pairs[:3])

    return None


def _compute_acf(x, nlags=None):
    n = len(x)
    if nlags is None:
        nlags = min(n // 2, 500)
    nlags = min(nlags, n // 2)

    x_centered = x - np.mean(x)
    var = np.var(x_centered)
    if var < 1e-12:
        return None

    acf = np.zeros(nlags + 1)
    for k in range(nlags + 1):
        acf[k] = np.sum(x_centered[: n - k] * x_centered[k:]) / (n * var)

    return acf


def compute_metrics_comparison(loops_data, selected_indices):
    if not selected_indices or len(selected_indices) < 2:
        return None

    all_metrics = {}
    metric_names = ["IAE", "ISE", "ITAE", "overshoot_pct", "settling_time",
                    "steady_state_error", "oscillation_period", "decay_ratio"]

    for idx in selected_indices:
        loop = loops_data[idx]
        metrics = compute_metrics(loop["pv"], loop["sp"], loop["co"], loop["sampling_period"])
        all_metrics[loop["name"]] = metrics

    return all_metrics, metric_names
