"""
提取QRS电轴数据
"""
import math
import numpy as np


def frontal_axes(lead_i, lead_avf, bounds, r_index, fs):
    """Estimate P, QRS and T axes from synchronized, calibrated beat templates."""
    templates = np.asarray([lead_i, lead_avf], dtype=float)
    axes = {}
    qrs_on = bounds['ECG_R_Onset']
    qrs_off = bounds['ECG_R_Offset']
    if not (0 < qrs_on <= r_index < qrs_off < templates.shape[1]):
        return axes

    # Use the quiet part of the PR segment as a common time reference.
    baseline_end = qrs_on - max(1, round(0.008 * fs))
    baseline_start = max(0, baseline_end - max(2, round(0.03 * fs)))
    p_off = bounds['ECG_P_Offset']
    if np.isfinite(p_off) and p_off < qrs_on:
        baseline_start = max(baseline_start, int(p_off))
    if baseline_end - baseline_start < 2:
        return axes
    baseline = np.median(templates[:, baseline_start:baseline_end], axis=1)

    intervals = (
        ('P', 'P', bounds['ECG_P_Peak'], 0, qrs_on),
        ('QRS', 'R', r_index, 0, templates.shape[1] - 1),
        ('T', 'T', bounds['ECG_T_Peak'], qrs_off, templates.shape[1] - 1),
    )
    for name, marker, peak, lower, upper in intervals:
        start, end = (bounds[f'ECG_{marker}_{edge}'] for edge in ('Onset', 'Offset'))
        if not (lower <= start <= peak < end <= upper and end - start >= 2):
            continue
        segment = templates[:, int(start):int(end) + 1] - baseline[:, None]
        # Signed trapezoidal area divided by interval length. The common
        # interval length cancels in the angle and keeps the magnitude in mV.
        horizontal, inferior = (segment[:, :-1] + segment[:, 1:]).mean(axis=1) / 2
        inferior *= 2 / math.sqrt(3)
        # A nearly zero resultant has no stable direction (5 µV guard).
        if math.hypot(horizontal, inferior) >= 0.005:
            axes[name] = math.degrees(math.atan2(inferior, horizontal))
    return axes


def calculate_axis_using_i_iii_lead(q_r_s_amplitude_i, q_r_s_amplitude_iii):
    """
    通过I和III导联的Q、R、S的振幅和，计算QRS电轴

    Parameters
    ----------
    q_r_s_amplitude_i : I导联上Q、R、S的振幅和
    q_r_s_amplitude_iii : III导联上Q、R、S的振幅和

    Returns
    -------
    电轴度数及偏转类型

    """
    # 电轴
    axis = None
    deviation_type = None
    # 估计avF导联电轴
    vector_avf = q_r_s_amplitude_iii + q_r_s_amplitude_i / 2
    # 排除±90度
    if q_r_s_amplitude_i == 0:
        if vector_avf > 0:
            axis = 90
            deviation_type = 'Normal'
        elif vector_avf < 0:
            axis = -90
            deviation_type = 'Left'
        return axis, deviation_type

    # 计算度数
    x = abs(2 / math.sqrt(3) * (vector_avf / q_r_s_amplitude_i))
    axis_tmp = math.degrees(math.atan(x))
    # 调整度数分区
    if q_r_s_amplitude_i > 0 and vector_avf > 0:
        axis = int(round(axis_tmp))
    elif q_r_s_amplitude_i > 0 and vector_avf < 0:
        axis = -int(round(axis_tmp))
    elif q_r_s_amplitude_i < 0 and vector_avf > 0:
        axis = 180 - int(round(axis_tmp))
    elif q_r_s_amplitude_i < 0 and vector_avf < 0:
        axis = -180 + int(round(axis_tmp))
    if axis is not None:
        if -30 <= axis <= 90:
            deviation_type = 'Normal'
        elif 90 < axis <= 180:
            deviation_type = 'Right'
        elif -90 <= axis < -30:
            deviation_type = 'Left'
        elif -180 < axis < -90:
            deviation_type = 'Extreme'
    return axis, deviation_type


def calculate_q_r_s_amplitude(q_amplitude, r_amplitude, s_amplitude):
    """
    计算综合振幅

    Parameters
    ----------
    q_amplitude : Q波振幅
    r_amplitude : R波振幅
    s_amplitude : S波振幅

    Returns
    -------
    综合振幅

    """
    if q_amplitude is None:
        q_amplitude = 0

    if r_amplitude is None:
        r_amplitude = 0

    if s_amplitude is None:
        s_amplitude = 0

    return r_amplitude - q_amplitude - s_amplitude
