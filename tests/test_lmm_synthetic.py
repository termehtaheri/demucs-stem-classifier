import numpy as np
import soundfile as sf
from lmm.mix_recon import reconstruct

def _sine(f, sr, dur, amp=0.1):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    return (amp*np.sin(2*np.pi*f*t)).astype(np.float32)

def test_reconstruct_synthetic(tmp_path):
    sr = 16000
    dur = 1.0
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()

    s1 = _sine(220, sr, dur, 0.2)
    s2 = _sine(440, sr, dur, 0.15)
    s3 = _sine(880, sr, dur, 0.1)

    sf.write(stems_dir / "s1.wav", s1, sr)
    sf.write(stems_dir / "s2.wav", s2, sr)
    sf.write(stems_dir / "s3.wav", s3, sr)

    master = 0.8*s1 + 0.6*s2 + 0.4*s3
    master_path = tmp_path / "master.wav"
    sf.write(master_path, master, sr)

    report = reconstruct(str(master_path), str(stems_dir), alpha=0.1)
    assert report["metrics"]["r2"] > 0.95
