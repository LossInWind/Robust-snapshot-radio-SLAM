from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .simulation_setup import simulation_setup
from .slam import localization, mapping, SlamState
from .los_detection import los_detection
from .result_summary import result_summary


def run() -> None:
    sim, params, _ = simulation_setup()

    T = int(params['T'])
    obj_list: List[Dict[str, Any]] = [None for _ in range(T)]

    for k in range(T):
        dt = np.zeros((3,), dtype=float)
        tx = sim['tx'][:, k]
        y = sim['y'][k]
        power = sim['power'][k]

        # 1st try LoS localization
        los_candidate = int(np.argmin(y[0, :])) if y.shape[1] > 0 else None

        obj, dt_loc = localization(tx, y, power, los_candidate, params)
        dt[0] = dt_loc

        if obj.xn is None or int(np.sum(obj.inliers)) == 2 or not los_detection(obj, tx, float(power[los_candidate] if los_candidate is not None else 0.0), params):
            # NLoS condition detected --> NLoS localization
            los_candidate = None
            obj, dt_loc = localization(tx, y, power, los_candidate, params)
            dt[1] = dt_loc
        else:
            dt[1] = 0.0

        # mapping
        obj, dt_map = mapping(obj, tx, y, power, params)
        dt[2] = dt_map

        # store
        obj_dict = {
            'xn': obj.xn,
            'xl': obj.xl,
            'L': obj.L,
            'inliers': obj.inliers,
            'los_candidate': obj.los_candidate,
            'dt': dt,
        }
        obj_list[k] = obj_dict

    result_summary(sim, obj_list, params)


if __name__ == '__main__':
    run()