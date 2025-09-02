from __future__ import annotations
import os
from typing import Dict, Tuple, List
import torch
import torchaudio
import pandas as pd
from demucs.pretrained import get_model
from demucs.apply import apply_model

SOURCES = ("vocals", "drums", "bass", "other")

def load_demucs(model_name: str = "htdemucs", device: str | torch.device | None = None):
    device = torch.device(device) if isinstance(device, str) else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = get_model(name=model_name).to(device)
    return model, device

def rms_energy(waveform: torch.Tensor) -> float:
    if waveform.dim() == 2 and waveform.size(0) == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    return torch.sqrt(torch.mean(waveform ** 2)).item()

def classify_stem(model, device, wav: torch.Tensor) -> Tuple[str, Dict[str, float], float]:
    wav = wav.to(device).unsqueeze(0)
    separated = apply_model(model, wav, split=True, overlap=0.25, progress=False, device=device)
    src_dict = dict(zip(model.sources, separated[0].cpu()))
    energies = {k: rms_energy(v) for k, v in src_dict.items()}
    ranked = sorted(energies.items(), key=lambda kv: kv[1], reverse=True)
    top_label = ranked[0][0]
    confidence = ranked[0][1] - ranked[1][1]
    return top_label, energies, confidence

def run_folder(stems_dir: str, out_csv: str, model_name: str = "htdemucs") -> pd.DataFrame:
    model, device = load_demucs(model_name)
    rows: List[Dict] = []
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(stems_dir, fname)
        wav, _ = torchaudio.load(path)
        if wav.shape[0] == 1:
            wav = torch.cat([wav, wav], dim=0)
        try:
            label, energy_map, conf = classify_stem(model, device, wav)
            rows.append({"filename": fname, "predicted_label": label, "confidence": conf, **energy_map})
        except Exception as e:
            row = {"filename": fname, "predicted_label": "ERROR", "confidence": 0.0}
            for s in SOURCES:
                row[s] = 0.0
            row["error"] = str(e)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def low_confidence(df: pd.DataFrame, quantile: float = 0.10) -> pd.DataFrame:
    thr = float(df["confidence"].quantile(quantile))
    return df[df["confidence"] < thr].copy()
