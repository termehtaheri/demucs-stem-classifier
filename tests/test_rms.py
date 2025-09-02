import torch
from mlsc.classifier import rms_energy

def test_rms_energy_monovs_stereo_same_scale():
    mono = torch.ones(1, 4800) * 0.1
    stereo = torch.cat([mono, mono], dim=0)
    assert abs(rms_energy(mono) - rms_energy(stereo)) < 1e-8
