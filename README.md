# Robust snapshot radio SLAM (Python port)

This repository contains a Python implementation of the robust snapshot radio SLAM algorithm presented in [1]. It is a faithful port of the original MATLAB version, using NumPy/SciPy/Matplotlib.

## Quickstart

- Create a virtual environment and install dependencies:

```
python3 -m venv .venv
. .venv/bin/activate
pip install -r python/requirements.txt
```

- Run the experiment:

```
python3 -m python.main
```

Expected output is similar to:

- LOS, estimates: 32/45, rmse: 0.2882 [m] 1.9456 [deg] 1.0554 [ns]
- NLOS, estimates: 13/45, rmse: 0.4886 [m] 2.2702 [deg] 2.1263 [ns]
- ALL, estimates: 45/45, rmse: 0.3578 [m] 2.0447 [deg] 1.4485 [ns]

Notes:
- Timing numbers will differ from MATLAB due to the lack of MEX acceleration. Consider Numba/Cython for further optimization.
- Input measurements are loaded from `measurements/Kampusareena_slam_measurements_svd_p999.mat`.

## Project structure

- `python/` – Python implementation
  - `main.py` – entry point
  - `slam.py` – core SLAM logic (`localization`, `mapping`)
  - `linear_algebra.py` – small linear algebra helpers
  - `simulation_setup.py` – data loading and preprocessing
  - `los_detection.py` – LoS detection
  - `result_summary.py` – metrics and plotting

## References

[1] Ossi Kaltiokallio, Elizaveta Rastorgueva-Foi, Jukka Talvitie, Yu Ge, Henk Wymeersch and Mikko Valkama, "Robust snapshot radio SLAM," accepted to IEEE Transactions on Vehicular Technology. [Online] Available: https://arxiv.org/abs/2404.10291
