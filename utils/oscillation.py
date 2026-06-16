import numpy as np
from scipy import signal


def detect_oscillation(pv, sp, co, ts):
    err = pv - sp
    err_c = err - np.mean(err)

    nlags = min(len(err_c) // 2, 500)
    acf = _compute_acf(err_c, nlags)

    is_oscillating = False
    osc_period = None
    osc_frequency = None
    osc_amplitude = None

    if acf is not None and len(acf) > 10:
        zero_crossings = []
        for i in range(1, len(acf)):
            if acf[i - 1] > 0 and acf[i] <= 0:
                zero_crossings.append(i)

        if len(zero_crossings) >= 2:
            periods = []
            for j in range(1, len(zero_crossings)):
                half_period_samples = zero_crossings[j] - zero_crossings[j - 1]
                periods.append(half_period_samples * 2 * ts)

            if len(periods) >= 1:
                avg_period = np.mean(periods[:3])
                peaks_acf, _ = signal.find_peaks(acf, distance=max(3, int(avg_period / ts * 0.5)))
                if len(peaks_acf) >= 2:
                    peak_heights = acf[peaks_acf]
                    if peak_heights[0] > 0.1:
                        is_oscillating = True
                        osc_period = avg_period
                        osc_frequency = 1.0 / avg_period if avg_period > 0 else 0
                        osc_amplitude = float(np.std(err_c) * np.sqrt(2))

    diagnosis = diagnose_root_cause(pv, sp, co, ts, is_oscillating, osc_period)

    return {
        "is_oscillating": is_oscillating,
        "period": osc_period,
        "frequency": osc_frequency,
        "amplitude": osc_amplitude,
        "acf": acf,
        "diagnosis": diagnosis,
    }


def diagnose_root_cause(pv, sp, co, ts, is_oscillating, osc_period):
    if not is_oscillating:
        return {
            "root_cause": "无振荡",
            "description": "未检测到明显振荡，控制回路运行平稳。",
            "recommendation": "继续监控。",
        }

    diagnosis = {
        "root_cause": "未知",
        "description": "",
        "recommendation": "",
    }

    err = pv - sp
    co_c = co - np.mean(co)

    co_saturation = _check_co_saturation(co)

    if co_saturation["saturated"]:
        diagnosis["root_cause"] = "执行器饱和/阀门粘滞"
        diagnosis["description"] = (
            f"CO信号存在明显的限幅现象：到达上限{co_saturation['upper_pct']:.1f}%和下限"
            f"{co_saturation['lower_pct']:.1f}%。"
            f"平顶时间占比{co_saturation['flat_ratio']:.1f}%。"
            "CO呈现平顶锯齿波形，说明执行器到达极限后反弹，"
            "这通常由阀门粘滞或执行器行程不足导致。"
        )
        diagnosis["recommendation"] = "建议检查阀门定位器、气源压力及执行机构。"
        return diagnosis

    phase_info = _check_phase_relationship(pv, sp, co, ts, osc_period)

    if phase_info["phase_locked"]:
        if phase_info["phase_diff"] < np.pi / 3:
            diagnosis["root_cause"] = "积分过强"
            diagnosis["description"] = (
                f"CO与PV同频振荡，相位差较小({np.degrees(phase_info['phase_diff']):.1f}°)，"
                "说明控制器对误差的积分作用过强，导致系统持续振荡。"
                "振荡周期为{:.1f}s。".format(osc_period if osc_period else 0)
            )
            diagnosis["recommendation"] = "建议减小积分增益Ki，或增加微分增益Kd。"
        else:
            diagnosis["root_cause"] = "控制器参数不当"
            diagnosis["description"] = (
                f"CO与PV同频振荡，相位差为{np.degrees(phase_info['phase_diff']):.1f}°，"
                "控制器参数可能不适合当前过程动态特性。"
            )
            diagnosis["recommendation"] = "建议重新辨识过程模型并整定PID参数。"
    else:
        diagnosis["root_cause"] = "外部扰动引起振荡"
        diagnosis["description"] = "振荡可能由外部周期性扰动或上游回路引起，非控制器参数问题。"
        diagnosis["recommendation"] = "建议排查上游扰动源，考虑增加前馈控制。"

    return diagnosis


def _compute_acf(x, nlags):
    n = len(x)
    nlags = min(nlags, n // 2)
    acf = np.zeros(nlags + 1)
    var = np.var(x)
    if var < 1e-12:
        return None
    for k in range(nlags + 1):
        acf[k] = np.sum(x[: n - k] * x[k:]) / (n * var)
    return acf


def _check_co_saturation(co, threshold=2.0):
    co_diff = np.abs(np.diff(co))
    flat_mask = co_diff < threshold

    flat_ratio = np.sum(flat_mask) / len(co_diff) * 100

    upper_limit = np.max(co)
    lower_limit = np.min(co)
    at_upper = np.sum(co >= upper_limit - 1.0) / len(co) * 100
    at_lower = np.sum(co <= lower_limit + 1.0) / len(co) * 100

    saturated = (at_upper > 5 or at_lower > 5) and flat_ratio > 50

    return {
        "saturated": saturated,
        "upper_pct": at_upper,
        "lower_pct": at_lower,
        "flat_ratio": flat_ratio,
    }


def _check_phase_relationship(pv, sp, co, ts, osc_period):
    if osc_period is None or osc_period <= 0:
        return {"phase_locked": False, "phase_diff": 0.0}

    err = pv - sp
    err_c = err - np.mean(err)
    co_c = co - np.mean(co)

    n = min(len(err_c), len(co_c))
    max_lag = min(int(osc_period * 2 / ts), n // 2)

    cross_corr = np.correlate(err_c[:n], co_c[:n], mode="full")
    mid = len(cross_corr) // 2
    cross_corr = cross_corr[mid - max_lag: mid + max_lag + 1]

    if len(cross_corr) == 0 or np.max(np.abs(cross_corr)) < 1e-10:
        return {"phase_locked": False, "phase_diff": 0.0}

    peak_idx = np.argmax(np.abs(cross_corr))
    peak_corr = cross_corr[peak_idx]
    max_corr = np.max(np.abs(cross_corr))
    peak_idx_norm = peak_idx / max(1, len(cross_corr) - 1)

    if max_corr > 0.3:
        lag_samples = peak_idx - max_lag
        phase_diff = abs(lag_samples * ts / osc_period) * 2 * np.pi
        phase_diff = phase_diff % np.pi

        return {
            "phase_locked": True,
            "phase_diff": float(phase_diff),
            "cross_correlation_max": float(max_corr),
        }

    return {"phase_locked": False, "phase_diff": 0.0}
