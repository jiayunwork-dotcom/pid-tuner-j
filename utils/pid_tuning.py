import numpy as np


def compute_pid_recommendations(K, T, L, controller_type="PI", lambda_val=None, imc_filter=None):
    results = {}

    results["ZN_general"] = _ziegler_nichols(K, T, L, controller_type, variant="general")
    results["ZN_no_overshoot"] = _ziegler_nichols(K, T, L, controller_type, variant="no_overshoot")
    results["ZN_quarter_decay"] = _ziegler_nichols(K, T, L, controller_type, variant="quarter_decay")
    results["Cohen_Coon"] = _cohen_coon(K, T, L, controller_type)
    results["Lambda"] = _lambda_tuning(K, T, L, controller_type, lambda_val)
    results["IMC"] = _imc_tuning(K, T, L, controller_type, imc_filter)
    results["ITAE_load"] = _itae_tuning(K, T, L, controller_type, variant="load")
    results["ITAE_setpoint"] = _itae_tuning(K, T, L, controller_type, variant="setpoint")

    return results


def _ziegler_nichols(K, T, L, controller_type, variant="general"):
    if L <= 0:
        L = 0.01
    r = L / T

    if variant == "general":
        if controller_type == "P":
            Kp = T / (K * L)
            Ki = 0
            Kd = 0
        elif controller_type == "PI":
            Kp = 0.9 * T / (K * L)
            Ki = Kp / (3.33 * L)
            Kd = 0
        elif controller_type == "PD":
            Kp = 1.2 * T / (K * L)
            Ki = 0
            Kd = Kp * 0.5 * L
        else:
            Kp = 1.2 * T / (K * L)
            Ki = Kp / (2.0 * L)
            Kd = Kp * 0.5 * L
    elif variant == "no_overshoot":
        if controller_type == "P":
            Kp = 0.5 * T / (K * L)
            Ki = 0
            Kd = 0
        elif controller_type == "PI":
            Kp = 0.45 * T / (K * L)
            Ki = Kp / (4.0 * L)
            Kd = 0
        else:
            Kp = 0.6 * T / (K * L)
            Ki = Kp / (3.0 * L)
            Kd = Kp * 0.5 * L
    elif variant == "quarter_decay":
        if controller_type == "P":
            Kp = T / (K * L)
            Ki = 0
            Kd = 0
        elif controller_type == "PI":
            Kp = 0.9 * T / (K * L)
            Ki = Kp / (3.33 * L)
            Kd = 0
        else:
            Kp = 1.2 * T / (K * L)
            Ki = Kp / (2.0 * L)
            Kd = Kp * 0.5 * L
    else:
        Kp = 1.2 * T / (K * L)
        Ki = Kp / (2.0 * L)
        Kd = Kp * 0.5 * L

    return {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd), "name": f"Z-N ({variant})"}


def _cohen_coon(K, T, L, controller_type):
    if L <= 0:
        L = 0.01
    r = L / T

    if controller_type == "P":
        Kp = (1 / K) * r ** (-1) * (1 + r / 3)
        Ki = 0
        Kd = 0
    elif controller_type == "PI":
        Kp = (1 / K) * (0.9 * r ** (-1) + 0.083)
        Ki = Kp / (L * (3.33 + 0.31 * r) / (1 + 2.2 * r))
        Kd = 0
    elif controller_type == "PD":
        Kp = (1 / K) * (1.24 * r ** (-1) + 0.16)
        Ki = 0
        Kd = Kp * L * (0.34 - 0.11 * r) / (1 + 0.18 * r)
    else:
        Kp = (1 / K) * (1.35 * r ** (-1) + 0.27)
        Ki = Kp / (L * (2.5 + 0.52 * r) / (1 + 0.63 * r))
        Kd = Kp * L * (0.37 - 0.11 * r) / (1 + 0.18 * r)

    return {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd), "name": "Cohen-Coon"}


def _lambda_tuning(K, T, L, controller_type, lambda_val=None):
    if lambda_val is None:
        lambda_val = max(T, 3 * L)

    Tf = lambda_val

    if controller_type == "P":
        Kp = T / (K * (Tf + L))
        Ki = 0
        Kd = 0
    elif controller_type == "PI":
        Kp = T / (K * (Tf + L))
        Ki = Kp / T
        Kd = 0
    else:
        Kp = (2 * T + L) / (K * (Tf + L))
        Ki = Kp / (T + L / 2)
        Kd = Kp * T * L / (2 * T + L)

    return {
        "Kp": float(Kp),
        "Ki": float(Ki),
        "Kd": float(Kd),
        "name": f"Lambda (τc={Tf:.1f}s)",
    }


def _imc_tuning(K, T, L, controller_type, imc_filter=None):
    if imc_filter is None:
        imc_filter = max(L, T * 0.1)

    Tf = imc_filter

    if controller_type == "P":
        Kp = T / (K * (Tf + L))
        Ki = 0
        Kd = 0
    elif controller_type == "PI":
        Kp = T / (K * (Tf + L))
        Ki = Kp / T
        Kd = 0
    else:
        Kp = (T + L / 2) / (K * (Tf + L))
        Ki = Kp / (T + L / 2)
        Kd = Kp * T * L / (2 * T + L)

    return {
        "Kp": float(Kp),
        "Ki": float(Ki),
        "Kd": float(Kd),
        "name": f"IMC (λ={Tf:.1f}s)",
    }


def _itae_tuning(K, T, L, controller_type, variant="load"):
    if L <= 0:
        L = 0.01
    r = L / T

    if variant == "load":
        if controller_type == "P":
            Kp = 0.49 / K * r ** (-1.084)
            Ki = 0
            Kd = 0
        elif controller_type == "PI":
            Kp = 0.859 / K * r ** (-0.977)
            Ti = T / 0.674 * r ** (-0.68)
            Ki = Kp / Ti
            Kd = 0
        else:
            Kp = 1.357 / K * r ** (-0.947)
            Ti = T / 0.938 * r ** (-0.485)
            Td = T * 0.381 * r ** (0.718)
            Ki = Kp / Ti
            Kd = Kp * Td
    else:
        if controller_type == "P":
            Kp = 0.49 / K * r ** (-1.084)
            Ki = 0
            Kd = 0
        elif controller_type == "PI":
            Kp = 0.586 / K * r ** (-0.916)
            Ti = T / 1.03 * r ** (-0.165)
            Ki = Kp / Ti
            Kd = 0
        else:
            Kp = 0.965 / K * r ** (-0.85)
            Ti = T / 0.796 * r ** (-0.147)
            Td = T * 0.308 * r ** (0.929)
            Ki = Kp / Ti
            Kd = Kp * Td

    return {
        "Kp": float(Kp),
        "Ki": float(Ki),
        "Kd": float(Kd),
        "name": f"ITAE ({variant})",
    }


def format_tuning_table(recommendations):
    rows = []
    for method, params in recommendations.items():
        rows.append({
            "方法": params["name"],
            "Kp": f"{params['Kp']:.4f}",
            "Ki": f"{params['Ki']:.4f}",
            "Kd": f"{params['Kd']:.4f}",
        })
    return rows
