import numpy as np


def simulate_closed_loop(K, T, L, Kp, Ki, Kd, ts, sim_time=600, sp_step_time=50, sp_step_size=10.0,
                          controller_type="PID", action_direction="反作用",
                          current_Kp=None, current_Ki=None, current_Kd=None):
    n = int(sim_time / ts)
    t = np.arange(n) * ts

    sp = np.zeros(n)
    sp[:int(sp_step_time / ts)] = 0
    sp[int(sp_step_time / ts):] = sp_step_size

    pv_new = _simulate_pid(K, T, L, Kp, Ki, Kd, sp, ts, n, controller_type, action_direction)

    pv_current = None
    if current_Kp is not None:
        pv_current = _simulate_pid(K, T, L, current_Kp, current_Ki or 0, current_Kd or 0,
                                    sp, ts, n, controller_type, action_direction)

    return t, sp, pv_new, pv_current


def _simulate_pid(K, T, L, Kp, Ki, Kd, sp, ts, n, controller_type, action_direction):
    pv = np.zeros(n)
    co = np.zeros(n)
    integral = 0.0
    prev_err = 0.0

    direction = 1.0 if action_direction == "反作用" else -1.0

    delay_samples = max(0, int(L / ts))
    co_history = np.zeros(n + delay_samples + 1)

    for i in range(1, n):
        err = sp[i] - pv[i - 1]

        integral += err * ts
        derivative = (err - prev_err) / ts

        if controller_type == "P":
            co_val = Kp * err
        elif controller_type == "PI":
            co_val = Kp * err + Ki * integral
        elif controller_type == "PD":
            co_val = Kp * err + Kd * derivative
        else:
            co_val = Kp * err + Ki * integral + Kd * derivative

        co_val *= direction
        co_val = np.clip(co_val, 0, 100)

        if (co_val <= 0 or co_val >= 100) and controller_type in ("PI", "PID"):
            if (err > 0 and co_val >= 100) or (err < 0 and co_val <= 0):
                pass
            else:
                integral -= err * ts

        co[i] = co_val

        delayed_idx = max(0, i - delay_samples)
        u = co[delayed_idx] if delayed_idx < n else co[delayed_idx - 1]

        dpv = (K * u / 100.0 - pv[i - 1]) / T * ts
        pv[i] = pv[i - 1] + dpv
        prev_err = err

    return pv
