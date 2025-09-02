# demucs-stem-classifier

**Training-free stem classification** into `{vocals, drums, bass, other}` using the pretrained **Demucs** separator.  
We separate each input stem and pick the label with the **highest RMS energy** among Demucs’ sources. Also outputs a **confidence** score (`top_energy - second_energy`) and flags low-confidence items.

> No proprietary data. Includes tiny synthetic stems for a quick demo.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install -e .

# (optional) create tiny demo stems
python scripts/make_demo_stems.py  # writes to data/samples/stems

# classify stems (replace with your folder of .wav stems)
python -m mlsc.cli \
  --stems_dir data/samples/stems \
  --out_csv classified_stems.csv \
  --lowconf_csv low_confidence.csv \
  --lowconf_quantile 0.10
```

## Outputs
- `classified_stems.csv`: `filename, predicted_label, confidence, vocals, drums, bass, other`
- `low_confidence.csv`: subset under your quantile threshold

## Why this works
Demucs is trained to split music into `{vocals, drums, bass, other}`.  
If a file is mostly “vocals-like,” its separated **vocals** track will have the highest energy.

## Roadmap
- Multi-label tagging via energy ratios  
- Chunk-wise voting for long stems  
- Optional tagger cross-check (HTSAT/PANNs)

## License & Attribution
- Code: MIT (see `LICENSE`)  
- Demucs: MIT by Meta/FAIR (see `LICENSES/demucs_LICENSE`)
