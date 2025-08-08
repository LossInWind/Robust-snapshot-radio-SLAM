from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .slam import SlamState


def los_detection(obj: Optional[SlamState], tx: np.ndarray, power: float, params: Dict) -> bool:
    if obj is None or obj.xn is None:
        return False
    theta = params['pathloss']['theta']
    threshold = float(params['pathloss']['threshold'])

    d = float(np.linalg.norm(obj.xn[0:2] - tx[0:2]))
    nu = (theta[0] + theta[1] * 10.0 * math.log10(d)) - power

    L = -0.5 * (math.log(2 * math.pi) + math.log(theta[2]) + (nu * nu) / theta[2])
    if L < threshold:
        return False
    return True