from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def result_summary(sim: Dict[str, Any], obj_list: List[Any], params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx = sim['tx']
    rx = sim['rx']

    T = int(params['T'])

    X = np.full((4, T), np.nan, dtype=float)
    M_cells: List[np.ndarray] = []
    LOS = -np.ones((T,), dtype=int)
    T_times = np.zeros((2, T), dtype=float)

    for k in range(T):
        obj = obj_list[k]
        if obj is None or obj['xn'] is None:
            continue
        X[:, k] = obj['xn']
        xl = obj.get('xl', None)
        if xl is not None and xl.size > 0:
            M_cells.append(xl)
        T_times[:, k] = np.array([obj['dt'][0] + obj['dt'][1], obj['dt'][2]])
        LOS[k] = 0 if (obj.get('los_candidate') is None) else 1

    K = int(np.sum(LOS != -1))
    N_arr = np.array([int(np.sum(LOS == 1)), int(np.sum(LOS == 0)), K], dtype=int)
    se = np.vstack([
        np.sum((rx[0:2, :] - X[0:2, :]) ** 2, axis=0),
        (rx[2:4, :] - X[2:4, :]) ** 2,
    ])

    RMSE = np.zeros((3, 3), dtype=float)
    for i in range(3):
        mask_LOS = (LOS == 1)
        mask_NLOS = (LOS == 0)
        mask_ALL = (LOS != -1)
        RMSE[i, 0] = math.sqrt(np.nanmean(se[i, mask_LOS])) if np.any(mask_LOS) else math.nan
        RMSE[i, 1] = math.sqrt(np.nanmean(se[i, mask_NLOS])) if np.any(mask_NLOS) else math.nan
        RMSE[i, 2] = math.sqrt(np.nanmean(se[i, mask_ALL])) if np.any(mask_ALL) else math.nan

    TIME_localization = np.array([
        np.nanmean(T_times[0, LOS == 1]),
        np.nanmean(T_times[0, LOS == 0]),
        np.nanmean(T_times[0, LOS != -1]),
    ])
    TIME_mapping = np.array([
        np.nanmean(T_times[1, LOS == 1]),
        np.nanmean(T_times[1, LOS == 0]),
        np.nanmean(T_times[1, LOS != -1]),
    ])

    c = 299792458.0
    labels = ['LOS', 'NLOS', 'ALL']
    for i in range(3):
        print(f"{labels[i]:>4s}, estimates: {N_arr[i]}/{K}, rmse: {RMSE[0,i]:.4f} [m] {RMSE[1,i]*180/math.pi:.4f} [deg] {RMSE[2,i]*1e9/c:.4f} [ns], time: {TIME_localization[i]*1000:.4f}/{TIME_mapping[i]*1000:.4f} [ms]")

    if params.get('PLOT_ON', False):
        x_lim = (-7, 11)
        y_lim = (-12, 4)

        plt.figure(100)
        plt.clf()
        ax = plt.gca()
        ax.set_box_aspect(1)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')

        layout = mpimg.imread('/workspace/measurements/kampusarena_depleted_background_medium.png')
        Ny, Nx = layout.shape[0], layout.shape[1]
        pixel_size_y = 16 / Ny
        pixel_size_x = 18 / Nx
        layout = layout[20:Ny - 20, 20:Nx - 20, :]

        x_ = (np.arange(1, Nx + 1)) * pixel_size_x
        y_ = (np.arange(1, Ny + 1)) * pixel_size_y
        x_ = x_ - 7
        y_ = y_ - 12
        y_ = y_[::-1]
        ax.imshow(layout, extent=[x_[0], x_[-1], y_[0], y_[-1]], alpha=0.75)

        ax.plot(tx[0, 0], tx[1, 0], 'bv', markersize=8, linewidth=2)
        ax.plot(rx[0, :], rx[1, :], 'b-', linewidth=2)

        if M_cells:
            mu = np.hstack(M_cells)
            ax.plot(mu[0, :], mu[1, :], 'r+', markersize=6, linewidth=1)
        ax.plot(X[0, :], X[1, :], 'kx', markersize=6, linewidth=1.5)
        for k in range(K):
            ax.plot([rx[0, k], X[0, k]], [rx[1, k], X[1, k]], 'k:', linewidth=1)

        plt.tight_layout()
        # Do not show plot in non-interactive environment

    return RMSE, TIME_mapping, se