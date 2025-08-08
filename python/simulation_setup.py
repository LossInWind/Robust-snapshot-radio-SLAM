from __future__ import annotations

import math
from typing import Dict, Tuple, List, Any

import numpy as np
from scipy.io import loadmat


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def simulation_setup(mat_path: str = '/workspace/measurements/Kampusareena_slam_measurements_svd_p999.mat') -> Tuple[Dict[str, Any], Dict[str, Any], List[dict]]:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    sim_m = data['sim']

    # Extract fields
    tx = np.array(getattr(sim_m, 'tx'), dtype=float)
    rx = np.array(getattr(sim_m, 'rx'), dtype=float)
    y_cells = getattr(sim_m, 'y')
    power_cells = getattr(sim_m, 'power')

    # Ensure lists of np.ndarray
    if isinstance(y_cells, np.ndarray) and y_cells.dtype == object:
        y_list = [np.array(y_cells[i], dtype=float) for i in range(y_cells.shape[0])]
    elif isinstance(y_cells, list):
        y_list = [np.array(v, dtype=float) for v in y_cells]
    else:
        # single measurement
        y_list = [np.array(y_cells, dtype=float)]

    if isinstance(power_cells, np.ndarray) and power_cells.dtype == object:
        power_list = [np.array(power_cells[i], dtype=float).reshape(-1) for i in range(power_cells.shape[0])]
    elif isinstance(power_cells, list):
        power_list = [np.array(v, dtype=float).reshape(-1) for v in power_cells]
    else:
        power_list = [np.array(power_cells, dtype=float).reshape(-1)]

    params: Dict[str, Any] = {}
    params['xn_dim'] = 4
    params['xl_dim'] = 2
    params['h_dim'] = 3
    c = 299792458.0
    params['R'] = np.diag([(1e-9 * c) ** 2, (3 * math.pi / 180.0) ** 2, (3 * math.pi / 180.0) ** 2])
    T = y_list.__len__()
    params['T'] = T
    params['bias'] = 10.0
    params['sigma2_bias'] = 1.0
    params['convergence_threshold'] = 1e-1
    params['max_iterations'] = 10
    params['pathloss'] = {
        'theta': np.array([-13.0, -1.7, 1.8 ** 2], dtype=float),
        'threshold': -10.8,
    }
    params['PLOT_ON'] = True
    params['MEX'] = False
    params['N'] = 361
    params['tx_offset'] = -1.6 * math.pi / 180.0
    params['epsilon'] = 0.1
    params['eta2_threshold'] = 0.1

    # Extend rx with bias row and calibrate BS
    bias = float(params['bias'])
    rx_ext = np.vstack([rx, np.zeros((1, T), dtype=float)])

    for k in range(T):
        if k < T - 1:
            bias = bias + math.sqrt(params['sigma2_bias']) * np.random.randn()
        rx_ext[3, k] = bias

        tx[2, k] = tx[2, k] + params['tx_offset']

        yk = y_list[k].copy()
        yk[0, :] = yk[0, :] + rx_ext[3, k]
        yk[1, :] = wrap_angle(yk[1, :])
        yk[2, :] = wrap_angle(yk[2, :])
        y_list[k] = yk

    sim: Dict[str, Any] = {
        'tx': tx,
        'rx': rx_ext,
        'y': y_list,
        'power': power_list,
    }

    obj_list: List[dict] = [None for _ in range(T)]
    return sim, params, obj_list