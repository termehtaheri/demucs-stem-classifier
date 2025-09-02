import argparse
from mlsc.classifier import run_folder, low_confidence

def main():
    p = argparse.ArgumentParser(description="Unsupervised stem classification via Demucs")
    p.add_argument("--stems_dir", required=True, help="Folder with .wav stems")
    p.add_argument("--out_csv", default="classified_stems.csv")
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--lowconf_csv", default="low_confidence.csv")
    p.add_argument("--lowconf_quantile", type=float, default=0.10)
    args = p.parse_args()

    df = run_folder(args.stems_dir, args.out_csv, model_name=args.model)
    if "confidence" in df.columns and not df.empty:
        low = low_confidence(df, args.lowconf_quantile)
        if not low.empty:
            low[["filename", "predicted_label", "confidence"]].to_csv(args.lowconf_csv, index=False)
            print(f"[info] Saved {len(low)} low-confidence items to {args.lowconf_csv}")

if __name__ == "__main__":
    main()
