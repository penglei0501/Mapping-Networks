from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-dimension sweeps for Projection Efficient CNN.")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/latent_sweep"))
    parser.add_argument("--summary", type=Path, default=Path("results/latent_sweep_summary.md"))
    return parser.parse_args()


def run_one(args: argparse.Namespace, latent_dim: int) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_ld{latent_dim}_e{args.epochs}.json"
    command = [
        sys.executable,
        "experiments/train_digits.py",
        "--model",
        "projection-efficient-cnn",
        "--layerwise",
        "--latent-dim",
        str(latent_dim),
        "--activation",
        "identity",
        "--output-gain",
        "3",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--dataset",
        args.dataset,
        "--output",
        str(output),
        "--no-progress",
    ]
    if args.quick:
        command.append("--quick")

    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def summarize(results: list[dict], summary_path: Path) -> None:
    rows = []
    for result in results:
        best = max(result["history"], key=lambda row: row["test_acc"])
        last = result["history"][-1]
        rows.append(
            {
                "latent_dim": result["args"]["latent_dim"],
                "trainable": result["trainable_parameters"],
                "target": result["target_parameters"],
                "epochs": result["args"]["epochs"],
                "best_epoch": best["epoch"],
                "best_acc": best["test_acc"],
                "final_acc": last["test_acc"],
                "elapsed_seconds": result["elapsed_seconds"],
                "seconds_per_epoch": result["seconds_per_epoch"],
            }
        )

    lines = [
        "| latent_dim | trainable params | target params | epochs | best epoch | best acc | final acc | total time | sec/epoch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {latent_dim} | {trainable:,} | {target:,} | {epochs} | {best_epoch} | "
            "{best_acc:.2%} | {final_acc:.2%} | {elapsed_seconds:.1f}s | {seconds_per_epoch:.1f}s |".format(
                **row
            )
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    results = []
    for latent_dim in args.latent_dims:
        try:
            results.append(run_one(args, latent_dim))
        except KeyboardInterrupt:
            print("\nInterrupted. Summarizing completed runs...", file=sys.stderr)
            break
    summarize(results, args.summary)


if __name__ == "__main__":
    main()
