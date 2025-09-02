from __future__ import annotations
import os, json
from typing import Dict, Tuple, List
import numpy as np
import soundfile as sf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

def load_audio(path: str):
    wav, sr = sf.read(path, always_2d=True)  # (T, C)
    if wav.shape[1] > 1:
        wav = wav.mean(axis=1, keepdims=True)
    return wav.reshape(-1), sr

def load_stems(stems_dir: str) -> Dict[str, Tuple[np.ndarray, int]]:
    stems = {}
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(stems_dir, fname)
        wav, sr = load_audio(path)
        stems[fname] = (wav, sr)
    if not stems:
        raise ValueError("No WAV stems found in directory.")
    return stems

def align_lengths(arrays: List[np.ndarray]) -> List[np.ndarray]:
    min_len = min(a.shape[0] for a in arrays)
    return [a[:min_len] for a in arrays]

def reconstruct(master_path: str, stems_dir: str, alpha: float = 1.0, report_path: str | None = None, save_error_wav: str | None = None) -> Dict:
    y, sr_master = load_audio(master_path)
    stems = load_stems(stems_dir)

    X_list, names, srs = [], [], []
    for name, (s, sr) in stems.items():
        srs.append(sr)
        names.append(name)
        X_list.append(s.astype(np.float64))

    if any(sr != sr_master for sr in srs):
        raise ValueError("All stems must match master's sample rate.")

    arrays = align_lengths([y] + X_list)
    y = arrays[0]
    X = np.stack(arrays[1:], axis=1)  # (T, S)

    norms = np.linalg.norm(X, axis=0) + 1e-12
    Xn = X / norms

    reg = Ridge(alpha=alpha, fit_intercept=False, positive=False)
    reg.fit(Xn, y)

    y_hat = reg.predict(Xn)
    r2 = float(r2_score(y, y_hat))
    mse = float(mean_squared_error(y, y_hat))
    gains = (reg.coef_ / norms).tolist()

    report = {
        "sample_rate": sr_master,
        "alpha": alpha,
        "metrics": {"r2": r2, "mse": mse},
        "stems": [{"name": n, "gain": g} for n, g in zip(names, gains)],
    }

    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    if save_error_wav:
        err = (y - y_hat).astype(np.float32)
        sf.write(save_error_wav, err, sr_master)

    return report
