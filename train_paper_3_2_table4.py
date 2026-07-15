from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_paper_3_1_1 import (
    ChunkedFixedProjector,
    FixedMappingLossCoefficients,
    TrainableMappingLossCoefficients,
    count_trainable_params,
    ensure_dir,
    resolve_device,
    seed_everything,
)


PAPER_TABLE_4_MSE = {
    ("baseline", None): 0.0035,
    ("mapping", 64): 0.0019,
    ("mapping", 256): 0.00093,
    ("mapping", 512): 0.00080,
    ("mapping", 1024): 0.00065,
    ("mapping", 2048): 0.00061,
}

MAPPING_LOSS_TERMS = ("stability", "smoothness", "alignment")


def numel(shape: Iterable[int]) -> int:
    return math.prod(shape)


def lstm_param_specs(
    input_size: int = 67,
    hidden_size: int = 32,
) -> OrderedDict[str, Tuple[int, ...]]:
    return OrderedDict(
        [
            ("lstm.weight_ih_l0", (4 * hidden_size, input_size)),
            ("lstm.weight_hh_l0", (4 * hidden_size, hidden_size)),
            ("lstm.bias_ih_l0", (4 * hidden_size,)),
            ("lstm.bias_hh_l0", (4 * hidden_size,)),
            ("head.weight", (1, hidden_size)),
            ("head.bias", (1,)),
        ]
    )


def target_param_count(input_size: int = 67, hidden_size: int = 32) -> int:
    return sum(numel(shape) for shape in lstm_param_specs(input_size, hidden_size).values())


class DirectLSTMRegressor(nn.Module):
    """Single-layer LSTM followed by a scalar regression head."""

    def __init__(self, input_size: int = 67, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def functional_lstm_forward(
    x: Tensor,
    params: Dict[str, Tensor],
    hidden_size: int,
) -> Tensor:
    """Apply a one-layer PyTorch-compatible LSTM using generated parameters."""

    batch_size = x.shape[0]
    h = x.new_zeros((batch_size, hidden_size))
    c = x.new_zeros((batch_size, hidden_size))

    for step in range(x.shape[1]):
        gates = F.linear(
            x[:, step],
            params["lstm.weight_ih_l0"],
            params["lstm.bias_ih_l0"],
        ) + F.linear(
            h,
            params["lstm.weight_hh_l0"],
            params["lstm.bias_hh_l0"],
        )
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
        c = torch.sigmoid(forget_gate) * c + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
        h = torch.sigmoid(output_gate) * torch.tanh(c)

    return F.linear(h, params["head.weight"], params["head.bias"]).squeeze(-1)


class MappingLSTMRegressor(nn.Module):
    """Table 4 Ours*: one trainable latent vector generates every LSTM weight."""

    def __init__(
        self,
        latent_dim: int,
        input_size: int = 67,
        hidden_size: int = 32,
        seed: int = 42,
        chunk_size: int = 4096,
        activation: str = "identity",
        weight_scale: float = 1.0,
        modulation_scale: float = 0.01,
        latent_init_std: float = 1.0,
        projection_init: str = "orthogonal",
        modulation_reduction: str = "mean",
        parameter_scale_mode: str = "fan-in",
        projection_layout: str = "blockwise",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.parameter_scale_mode = parameter_scale_mode
        self.specs = lstm_param_specs(input_size, hidden_size)
        self.target_param_count = target_param_count(input_size, hidden_size)
        self.projector = ChunkedFixedProjector(
            latent_dim=latent_dim,
            out_dim=self.target_param_count,
            seed=seed,
            chunk_size=chunk_size,
            activation=activation,
            weight_scale=weight_scale,
            modulation_scale=modulation_scale,
            latent_init_std=latent_init_std,
            projection_init=projection_init,
            modulation_reduction=modulation_reduction,
            projection_layout=projection_layout,
        )

    def _scale_param(self, name: str, value: Tensor) -> Tensor:
        if self.parameter_scale_mode == "paper":
            return value
        if name.endswith("weight") or ".weight_" in name:
            return value / math.sqrt(value.shape[1])
        if name.endswith("bias") or ".bias_" in name:
            return value * 0.1
        return value

    def generate_params(self, perturb_scale: float = 0.0) -> OrderedDict[str, Tensor]:
        if perturb_scale:
            noise = torch.randn_like(self.projector.z) * perturb_scale
            flat = self.projector(self.projector.z + noise)
        else:
            flat = self.projector()

        params: OrderedDict[str, Tensor] = OrderedDict()
        offset = 0
        for name, shape in self.specs.items():
            size = numel(shape)
            value = flat[offset:offset + size].view(shape)
            params[name] = self._scale_param(name, value)
            offset += size
        return params

    def forward(self, x: Tensor) -> Tensor:
        return functional_lstm_forward(x, self.generate_params(), self.hidden_size)

    def smoothness_loss(self) -> Tensor:
        return self.projector.smoothness_loss()

    def alignment_loss(self) -> Tensor:
        return self.projector.alignment_loss()


def load_npz_datasets(
    path: Path,
    input_size: int,
) -> Tuple[TensorDataset, TensorDataset, dict]:
    """Load the explicit Table 4 preprocessing contract.

    Required arrays are x_train, y_train, x_test, and y_test. Inputs must be
    [samples, time_steps, features]; targets must be [samples] or [samples, 1].
    """

    if not path.is_file():
        raise FileNotFoundError(f"Table 4 data file does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {"x_train", "y_train", "x_test", "y_test"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Missing arrays in {path}: {missing}")
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in required}

    x_train = torch.from_numpy(arrays["x_train"])
    x_test = torch.from_numpy(arrays["x_test"])
    y_train = torch.from_numpy(arrays["y_train"]).squeeze(-1)
    y_test = torch.from_numpy(arrays["y_test"]).squeeze(-1)

    for split, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
        if x.ndim != 3:
            raise ValueError(f"x_{split} must have shape [N, T, F], got {tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError(f"y_{split} must have shape [N] or [N, 1], got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x_{split} and y_{split} have different sample counts")
        if x.shape[2] != input_size:
            raise ValueError(
                f"x_{split} has {x.shape[2]} features; architecture requires {input_size}"
            )
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"{split} arrays contain NaN or infinity")

    metadata = {
        "data_file": str(path.resolve()),
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_test_shape": list(y_test.shape),
    }
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test), metadata


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, dict]:
    train_set, test_set, metadata = load_npz_datasets(Path(args.data_file), args.input_size)
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader, metadata


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = model(x)
        squared_error += F.mse_loss(prediction, y, reduction="sum").item()
        count += y.numel()
    if count == 0:
        raise ValueError("Evaluation loader produced no samples")
    return squared_error / count


def make_loss_coefficients(args: argparse.Namespace) -> Optional[nn.Module]:
    if args.mode != "mapping" or args.loss_mode == "task":
        return None
    cls = (
        TrainableMappingLossCoefficients
        if args.loss_coefficient_mode == "trainable"
        else FixedMappingLossCoefficients
    )
    return cls(
        args.lambda_stability,
        args.lambda_smoothness,
        args.lambda_alignment,
        enabled_terms=args.mapping_loss_terms,
    )


def mapping_batch_loss(
    args: argparse.Namespace,
    model: MappingLSTMRegressor,
    coefficients: Optional[nn.Module],
    x: Tensor,
    y: Tensor,
) -> Tuple[Tensor, dict]:
    prediction = model(x)
    task_loss = F.mse_loss(prediction, y)
    zero = task_loss.new_zeros(())

    if args.loss_mode == "task":
        return task_loss, {
            "task": float(task_loss.detach()),
            "stability": 0.0,
            "smoothness": 0.0,
            "alignment": 0.0,
        }

    if coefficients is None:
        raise RuntimeError("Full Mapping Loss requires coefficients")

    selected = set(args.mapping_loss_terms)
    if "stability" in selected:
        perturbed = functional_lstm_forward(
            x,
            model.generate_params(args.stability_sigma),
            model.hidden_size,
        )
        stability = F.mse_loss(perturbed, prediction)
    else:
        stability = zero
    smoothness = model.smoothness_loss() if "smoothness" in selected else zero
    alignment = model.alignment_loss() if "alignment" in selected else zero
    lambdas = coefficients()
    loss = (
        task_loss
        + lambdas["stability"] * stability
        + lambdas["smoothness"] * smoothness
        + lambdas["alignment"] * alignment
    )
    return loss, {
        "task": float(task_loss.detach()),
        "stability": float(stability.detach()),
        "smoothness": float(smoothness.detach()),
        "alignment": float(alignment.detach()),
    }


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_loader, test_loader, data_metadata = build_loaders(args)

    if args.mode == "baseline":
        model: nn.Module = DirectLSTMRegressor(args.input_size, args.hidden_size)
    else:
        if args.latent_dim is None:
            raise ValueError("Mapping mode requires --latent-dim")
        model = MappingLSTMRegressor(
            latent_dim=args.latent_dim,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            seed=args.seed,
            chunk_size=args.chunk_size,
            activation=args.activation,
            weight_scale=args.weight_scale,
            modulation_scale=args.modulation_scale,
            latent_init_std=args.latent_init_std,
            projection_init=args.projection_init,
            modulation_reduction=args.modulation_reduction,
            parameter_scale_mode=args.parameter_scale_mode,
            projection_layout=args.projection_layout,
        )
    model.to(device)

    coefficients = make_loss_coefficients(args)
    if coefficients is not None:
        coefficients.to(device)
    optimized = list(model.parameters())
    if coefficients is not None:
        optimized.extend(coefficients.parameters())
    optimizer = torch.optim.Adam(optimized, lr=args.lr, weight_decay=args.weight_decay)

    latent_label = f"_z{args.latent_dim}" if args.mode == "mapping" else ""
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.mode}_lstm_air_pollution{latent_label}_seed{args.seed}"
    run_dir = ensure_dir(Path(args.output_dir) / run_name)
    paper_mse = PAPER_TABLE_4_MSE.get((args.mode, args.latent_dim))
    generated_count = target_param_count(args.input_size, args.hidden_size)

    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "device_resolved": str(device),
            "paper_mse": paper_mse,
            "target_param_count": generated_count,
            "trainable_model_params": count_trainable_params(model),
            "data": data_metadata,
            "architecture_evidence": (
                "input_size=67 and hidden_size=32 are uniquely inferred from the "
                "paper's 12,961 parameter count under a one-layer LSTM plus scalar head"
            ),
            "paper_undisclosed": [
                "dataset file or URL",
                "feature construction",
                "sequence length",
                "train/test split",
                "normalization",
                "optimizer and training hyperparameters",
            ],
        }
    )
    save_json(config, run_dir / "config.json")

    initial_mse = evaluate(model, test_loader, device, args.max_test_batches)
    print("=" * 80)
    print("Paper 3.2 Table 4 Mapping LSTM reproduction")
    print("=" * 80)
    print(f"run_dir                 = {run_dir}")
    print(f"device                  = {device}")
    print(f"mode                    = {args.mode}")
    print(f"input_size/hidden_size  = {args.input_size}/{args.hidden_size}")
    print(f"trainable_model_params  = {count_trainable_params(model):,}")
    print(f"target_generated_params = {generated_count:,}")
    print(f"paper_mse               = {paper_mse}")
    print(f"initial_test_mse        = {initial_mse:.8f}")
    print("=" * 80)

    best_mse = math.inf
    best_epoch = 0
    history_path = run_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        started = time.time()
        for batch_index, (x, y) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "mapping":
                loss, _ = mapping_batch_loss(args, model, coefficients, x, y)
            else:
                loss = F.mse_loss(model(x), y)
            loss.backward()
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(optimized, args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_count += y.numel()

        test_mse = evaluate(model, test_loader, device, args.max_test_batches)
        if test_mse < best_mse:
            best_mse = test_mse
            best_epoch = epoch
            checkpoint = {"model": model.state_dict(), "epoch": epoch, "test_mse": test_mse}
            if coefficients is not None:
                checkpoint["loss_coefficients"] = coefficients.state_dict()
            torch.save(checkpoint, run_dir / "best_model.pt")

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "test_mse": test_mse,
            "best_test_mse": best_mse,
            "seconds": time.time() - started,
        }
        if coefficients is not None:
            row["mapping_loss_coefficients"] = coefficients.detached_values()
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.8f} "
            f"test_mse={test_mse:.8f} best={best_mse:.8f} "
            f"seconds={row['seconds']:.1f}"
        )

    final = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_test_mse": best_mse,
        "paper_mse": paper_mse,
        "gap_best_minus_paper": None if paper_mse is None else best_mse - paper_mse,
        "trainable_model_params": count_trainable_params(model),
        "trainable_loss_coefficients": 0 if coefficients is None else count_trainable_params(coefficients),
        "target_param_count": generated_count,
        "final_mapping_loss_coefficients": (
            None if coefficients is None else coefficients.detached_values()
        ),
    }
    save_json(final, run_dir / "final_result.json")
    print("=" * 80)
    print(f"best_epoch    = {best_epoch}")
    print(f"best_test_mse = {best_mse:.8f}")
    print(f"saved to      = {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Mapping Networks paper section 3.2, Table 4"
    )
    parser.add_argument("--mode", choices=["baseline", "mapping"], required=True)
    parser.add_argument("--data-file", required=True, help="Preprocessed .npz sequence dataset")
    parser.add_argument("--output-dir", default="results/table4")
    parser.add_argument("--input-size", type=int, default=67)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)

    parser.add_argument("--projection-init", choices=["orthogonal", "gaussian"], default="orthogonal")
    parser.add_argument("--projection-layout", choices=["global", "blockwise"], default="blockwise")
    parser.add_argument("--activation", choices=["identity", "tanh", "sin"], default="identity")
    parser.add_argument("--modulation-reduction", choices=["sum", "mean"], default="mean")
    parser.add_argument("--parameter-scale-mode", choices=["paper", "fan-in"], default="fan-in")
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--modulation-scale", type=float, default=0.01)
    parser.add_argument("--latent-init-std", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=4096)

    parser.add_argument("--loss-mode", choices=["task", "full"], default="task")
    parser.add_argument(
        "--mapping-loss-terms",
        nargs="+",
        choices=MAPPING_LOSS_TERMS,
        default=list(MAPPING_LOSS_TERMS),
    )
    parser.add_argument("--loss-coefficient-mode", choices=["fixed", "trainable"], default="trainable")
    parser.add_argument("--lambda-stability", type=float, default=0.05)
    parser.add_argument("--lambda-smoothness", type=float, default=5e-6)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    parser.add_argument("--stability-sigma", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "baseline" and args.loss_mode != "task":
        raise ValueError("Baseline mode supports task MSE only")
    if args.input_size == 67 and args.hidden_size == 32:
        assert target_param_count(args.input_size, args.hidden_size) == 12_961
    run(args)


if __name__ == "__main__":
    main()
