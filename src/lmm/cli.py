import argparse, json
from lmm.mix_recon import reconstruct

def main():
    p = argparse.ArgumentParser(description="Mix reconstruction & reporting via ridge regression")
    p.add_argument("--mode", choices=["reconstruct"], default="reconstruct")
    p.add_argument("--master", required=True, help="Path to master WAV")
    p.add_argument("--stems_dir", required=True, help="Directory of stem WAVs")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--report", default="lmm_report.json")
    p.add_argument("--save_error_wav", default="")
    args = p.parse_args()

    report = reconstruct(
        master_path=args.master,
        stems_dir=args.stems_dir,
        alpha=args.alpha,
        report_path=args.report,
        save_error_wav=(args.save_error_wav or None)
    )
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
