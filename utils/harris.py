import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def harris_index(pv, sp, ts, controller_type="PI"):
    err = pv - sp
    err_centered = err - np.mean(err)

    actual_var = np.var(err_centered)
    if actual_var < 1e-12:
        return 1.0, 0.0, 0.0

    n = len(err_centered)
    if n > 500:
        step = n // 500
        err_downsampled = err_centered[::step]
        ts_downsampled = ts * step
    else:
        err_downsampled = err_centered
        ts_downsampled = ts

    d = 1
    best_order = (1, d, 1)
    ar_params = np.array([1.0, -0.5])
    ma_params = np.array([1.0, 0.3])
    sigma_e = actual_var * 0.1

    try:
        model = ARIMA(err_downsampled, order=(1, 1, 1))
        res = model.fit(method="statespace", maxiter=50)
        ar_params = res.polynomial_ar
        ma_params = res.polynomial_ma
        sigma_e = np.var(res.resid)
        best_order = (1, d, 1)
    except Exception:
        try:
            model = ARIMA(err_downsampled, order=(1, 1, 0))
            res = model.fit(method="statespace", maxiter=50)
            ar_params = res.polynomial_ar
            ma_params = np.array([1.0])
            sigma_e = np.var(res.resid)
            best_order = (1, d, 0)
        except Exception:
            pass

    delay_samples = max(1, int(_estimate_delay(err, ts) / ts))

    psi_weights = _compute_psi_weights(ar_params, ma_params, best_order[1], delay_samples)

    mv_sum = 0.0
    for k in range(delay_samples):
        if k < len(psi_weights):
            mv_sum += psi_weights[k] ** 2

    min_var = sigma_e * mv_sum
    if actual_var > 0:
        harris = min_var / actual_var
    else:
        harris = 1.0

    harris = np.clip(harris, 0.0, 1.0)
    potential_improvement = (1.0 - harris) * 100.0

    return float(harris), float(potential_improvement), float(actual_var)


def _estimate_delay(err, ts):
    n = len(err)
    err_c = err - np.mean(err)

    max_lag = min(n // 3, 200)
    acf = np.zeros(max_lag)
    var = np.var(err_c)
    if var < 1e-12:
        return 1.0

    for k in range(max_lag):
        acf[k] = np.sum(err_c[: n - k] * err_c[k:]) / (n * var)

    for k in range(1, max_lag - 1):
        if acf[k] < 0 and k > 1:
            return k * ts

    return 1.0


def _compute_psi_weights(ar_poly, ma_poly, d, n_psi):
    n = max(n_psi * 2, 50)
    psi = np.zeros(n)
    psi[0] = 1.0

    ar_coeffs = -ar_poly[1:] if len(ar_poly) > 1 else np.array([])
    ma_coeffs = ma_poly[1:] if len(ma_poly) > 1 else np.array([])

    for i in range(1, n):
        val = 0.0
        for j in range(len(ar_coeffs)):
            if i - j - 1 >= 0:
                val += ar_coeffs[j] * psi[i - j - 1]
        if i - 1 < len(ma_coeffs):
            val += ma_coeffs[i - 1]
        psi[i] = val

    if d > 0:
        for _ in range(d):
            psi_cumsum = np.zeros(n)
            psi_cumsum[0] = psi[0]
            for i in range(1, n):
                psi_cumsum[i] = psi_cumsum[i - 1] + psi[i]
            psi = psi_cumsum

    return psi


def harris_gauge_color(harris_val):
    if harris_val >= 0.8:
        return "#27ae60"
    elif harris_val >= 0.5:
        return "#f39c12"
    else:
        return "#e74c3c"


def harris_status_text(harris_val):
    if harris_val >= 0.8:
        return "优秀 - 控制器接近最优"
    elif harris_val >= 0.5:
        return "有提升空间 - 建议优化参数"
    else:
        return "急需调优 - 控制器性能差"
