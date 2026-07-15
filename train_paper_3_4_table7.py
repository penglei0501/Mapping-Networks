from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_paper_3_1_1 import (
    ChunkedFixedProjector,
    MAPPING_LOSS_TERMS,
    TrainableMappingLossCoefficients,
    append_jsonl,
    count_trainable_params,
    ensure_dir,
    functional_cnn_forward,
    get_loaders,
    get_param_specs,
    now_string,
    numel_from_shape,
    resolve_device,
    save_json,
    seed_everything,
    validate_mapping_loss_terms,
)


TARGET_PARAM_COUNT = 108_618

PAPER_TABLE_7 = {
    ("baseline", 108_618, "mnist"): 0.9869,
    ("baseline", 108_618, "fmnist"): 0.9040,
    ("full-dnn", 6_753_104, "mnist"): 0.9712,
    ("full-dnn", 6_753_104, "fmnist"): 0.9011,
    ("ours-no-wm", 1_024, "mnist"): 0.9562,
    ("ours-no-wm", 1_024, "fmnist"): 0.8651,
    ("ours-no-wm", 2_048, "mnist"): 0.9655,
    ("ours-no-wm", 2_048, "fmnist"): 0.8766,
    ("ours", 1_024, "mnist"): 0.9788,
    ("ours", 1_024, "fmnist"): 0.8949,
    ("ours", 2_048, "mnist"): 0.9866,
    ("ours", 2_048, "fmnist"): 0.9188,
    ("lv-wmap", 2_048, "mnist"): 0.9790,
    ("lv-wmap", 2_048, "fmnist"): 0.8930,
    ("lv-wmap", 4_096, "mnist"): 0.9848,
    ("lv-wmap", 4_096, "fmnist"): 0.9193,
    ("lv-full-dnn", 543_095, "mnist"): 0.9616,
    ("lv-full-dnn", 543_095, "fmnist"): 0.9011,
    ("lv-full-dnn", 1_629_285, "mnist"): 0.9760,
    ("lv-full-dnn", 1_629_285, "fmnist"): 0.9067,
}

PAPER_UNDISCLOSED_SETTINGS = (
    "Full DNN layer dimensions",
    "activation function",
    "modulation scale",
    "latent initialization",
    "epochs",
    "batch size",
    "learning rate",
    "optimizer",
    "Mapping Loss coefficient initialization",
)

VARIANTS = (
    "baseline",
    "full-dnn",
    "ours-no-wm",
    "ours",
    "lv-wmap",
    "lv-full-dnn",
)


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    table_params: int
    latent_dim: Optional[int]
    modulation_dim: Optional[int]
    paper_acc: float
    disclosed_by_paper: bool
    implementation_note: str


def resolve_variant_spec(
    variant: str,
    table_params: Optional[int],
    dataset: str,
) -> VariantSpec:
    allowed_budgets = {
        "baseline": (108_618,),
        "full-dnn": (6_753_104,),
        "ours-no-wm": (1_024, 2_048),
        "ours": (1_024, 2_048),
        "lv-wmap": (2_048, 4_096),
        "lv-full-dnn": (543_095, 1_629_285),
    }
    if variant not in allowed_budgets:
        raise ValueError(f"Unknown Table 7 variant: {variant}")

    budgets = allowed_budgets[variant]
    if table_params is None:
        if len(budgets) != 1:
            raise ValueError(
                f"{variant} requires --table-params; choose one of {budgets}"
            )
        table_params = budgets[0]
    if table_params not in budgets:
        raise ValueError(
            f"Paper Table 7 reports {variant} with params {budgets}; "
            f"got {table_params}"
        )

    latent_dim: Optional[int]
    modulation_dim: Optional[int] = None
    disclosed = True
    if variant in {"baseline", "full-dnn"}:
        latent_dim = None
    elif variant in {"ours", "ours-no-wm"}:
        latent_dim = table_params
    elif variant == "lv-wmap":
        latent_dim = table_params // 2
        modulation_dim = latent_dim
    else:
        latent_dim = {543_095: 5, 1_629_285: 15}[table_params]

    notes = {
        "baseline": "Directly train all CNN2 parameters.",
        "full-dnn": (
            "Paper says the latent is fixed and mapping weights are trainable, "
            "but does not disclose the Full DNN architecture."
        ),
        "ours-no-wm": (
            "Train z with fixed orthogonal mapping weights and alpha=0."
        ),
        "ours": (
            "Train z with fixed orthogonal mapping weights additively modulated by z."
        ),
        "lv-wmap": (
            "Infer equal-length trainable z and independent modulation vector m "
            "from the reported total parameter budget."
        ),
        "lv-full-dnn": (
            "Train z and a bias-free dense mapping matrix; d*108618+d exactly "
            "matches the two reported parameter counts."
        ),
    }
    if variant == "full-dnn":
        disclosed = False

    return VariantSpec(
        variant=variant,
        table_params=table_params,
        latent_dim=latent_dim,
        modulation_dim=modulation_dim,
        paper_acc=PAPER_TABLE_7[(variant, table_params, dataset)],
        disclosed_by_paper=disclosed,
        implementation_note=notes[variant],
    )


def activate(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return value
    if name == "tanh":
        return torch.tanh(value)
    if name == "sin":
        return torch.sin(value)
    raise ValueError(f"Unknown activation: {name}")


def activation_derivative(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return torch.ones_like(value)
    if name == "tanh":
        activated = torch.tanh(value)
        return 1.0 - activated * activated
    if name == "sin":
        return torch.cos(value)
    raise ValueError(f"Unknown activation: {name}")


def orthogonal_parameter(rows: int, columns: int, seed: int) -> nn.Parameter:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.empty(rows, columns, dtype=torch.float32)
    nn.init.orthogonal_(value, generator=generator)
    return nn.Parameter(value)


class PaperFixedProjector(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        seed: int,
        chunk_size: int,
        activation_name: str,
        weight_scale: float,
        modulation_scale: float,
        latent_init_std: float,
        projection_init: str,
        modulation_reduction: str,
        projection_layout: str,
    ) -> None:
        super().__init__()
        self.core = ChunkedFixedProjector(
            latent_dim=latent_dim,
            out_dim=out_dim,
            seed=seed,
            chunk_size=chunk_size,
            activation=activation_name,
            weight_scale=weight_scale,
            modulation_scale=modulation_scale,
            latent_init_std=latent_init_std,
            projection_init=projection_init,
            modulation_reduction=modulation_reduction,
            projection_layout=projection_layout,
        )

    @property
    def latent_vector(self) -> Tensor:
        return self.core.z

    def forward(self, latent_override: Optional[Tensor] = None) -> Tensor:
        return self.core(latent_override)

    def smoothness_loss(self) -> Tensor:
        return self.core.smoothness_loss()

    def alignment_loss(self) -> Tensor:
        return self.core.alignment_loss()


class SeparateModulationProjector(nn.Module):
    """LV + WMAP: z and an independent modulation vector m are trainable."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        seed: int,
        chunk_size: int,
        activation_name: str,
        weight_scale: float,
        modulation_scale: float,
        latent_init_std: float,
        projection_init: str,
        modulation_reduction: str,
        projection_layout: str,
    ) -> None:
        super().__init__()
        self.activation_name = activation_name
        self.weight_scale = float(weight_scale)
        self.modulation_scale = float(modulation_scale)
        self.modulation_reduction = modulation_reduction
        self.out_dim = int(out_dim)
        self.core = ChunkedFixedProjector(
            latent_dim=latent_dim,
            out_dim=out_dim,
            seed=seed,
            chunk_size=chunk_size,
            activation="identity",
            weight_scale=1.0,
            modulation_scale=0.0,
            latent_init_std=latent_init_std,
            projection_init=projection_init,
            modulation_reduction=modulation_reduction,
            projection_layout=projection_layout,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 7919)
        self.modulation = nn.Parameter(
            torch.randn(latent_dim, generator=generator) * latent_init_std
        )

    @property
    def latent_vector(self) -> Tensor:
        return self.core.z

    def _alpha(self) -> float:
        if self.modulation_reduction == "sum":
            return self.modulation_scale
        if self.modulation_reduction == "mean":
            return self.modulation_scale / self.latent_vector.numel()
        raise ValueError(
            f"Unknown modulation reduction: {self.modulation_reduction}"
        )

    def _raw(self, latent: Tensor) -> Tensor:
        weight = self.core.cached_W.to(device=latent.device, dtype=latent.dtype)
        bias = self.core.cached_b.to(device=latent.device, dtype=latent.dtype)
        return (
            torch.matmul(latent, weight)
            + self._alpha() * torch.dot(latent, self.modulation)
            + bias
        )

    def forward(self, latent_override: Optional[Tensor] = None) -> Tensor:
        latent = self.latent_vector if latent_override is None else latent_override
        return activate(self._raw(latent), self.activation_name) * self.weight_scale

    def smoothness_loss(self) -> Tensor:
        latent = self.latent_vector
        weight = self.core.cached_W.to(device=latent.device, dtype=latent.dtype)
        alpha = self._alpha()
        raw = self._raw(latent)
        derivative_sq = activation_derivative(raw, self.activation_name).square()
        modulation_norm_sq = torch.sum(self.modulation.square())
        cross = torch.matmul(self.modulation, weight)
        column_norm_sq = (
            torch.sum(weight.square(), dim=0)
            + 2.0 * alpha * cross
            + alpha * alpha * modulation_norm_sq
        )
        return self.weight_scale**2 * torch.sum(derivative_sq * column_norm_sq)

    def alignment_loss(self) -> Tensor:
        latent = self.latent_vector
        row_mean = self.core.cached_W_row_mean.to(
            device=latent.device,
            dtype=latent.dtype,
        ) + self._alpha() * self.modulation
        return 1.0 - F.cosine_similarity(
            latent.unsqueeze(0),
            row_mean.unsqueeze(0),
            dim=1,
        ).mean()


class TrainableLinearProjector(nn.Module):
    """LV + FullDNN: both z and the full bias-free mapping matrix train."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        seed: int,
        activation_name: str,
        weight_scale: float,
        latent_init_std: float,
        projection_init: str,
    ) -> None:
        super().__init__()
        self.activation_name = activation_name
        self.weight_scale = float(weight_scale)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.z = nn.Parameter(
            torch.randn(latent_dim, generator=generator) * latent_init_std
        )
        if projection_init == "orthogonal":
            self.mapping_weight = orthogonal_parameter(
                latent_dim, out_dim, seed + 1
            )
        elif projection_init == "gaussian":
            generator.manual_seed(seed + 1)
            value = torch.randn(
                latent_dim,
                out_dim,
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(latent_dim)
            self.mapping_weight = nn.Parameter(value)
        else:
            raise ValueError(
                f"Unknown projection initialization: {projection_init}"
            )

    @property
    def latent_vector(self) -> Tensor:
        return self.z

    def _raw(self, latent: Tensor) -> Tensor:
        return torch.matmul(latent, self.mapping_weight)

    def forward(self, latent_override: Optional[Tensor] = None) -> Tensor:
        latent = self.z if latent_override is None else latent_override
        return activate(self._raw(latent), self.activation_name) * self.weight_scale

    def smoothness_loss(self) -> Tensor:
        raw = self._raw(self.z)
        derivative_sq = activation_derivative(raw, self.activation_name).square()
        column_norm_sq = torch.sum(self.mapping_weight.square(), dim=0)
        return self.weight_scale**2 * torch.sum(derivative_sq * column_norm_sq)

    def alignment_loss(self) -> Tensor:
        row_mean = self.mapping_weight.mean(dim=1)
        return 1.0 - F.cosine_similarity(
            self.z.unsqueeze(0),
            row_mean.unsqueeze(0),
            dim=1,
        ).mean()


class AssumedFullDNNProjector(nn.Module):
    """One explicit architecture matching 6,753,104; not disclosed by the paper."""

    INPUT_DIM = 307
    HIDDEN_DIM = 61

    def __init__(
        self,
        out_dim: int,
        seed: int,
        activation_name: str,
        latent_init_std: float,
    ) -> None:
        super().__init__()
        self.activation_name = activation_name
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.register_buffer(
            "fixed_z",
            torch.randn(self.INPUT_DIM, generator=generator) * latent_init_std,
        )
        self.weight1 = orthogonal_parameter(self.INPUT_DIM, self.HIDDEN_DIM, seed + 1)
        self.bias1 = nn.Parameter(torch.zeros(self.HIDDEN_DIM))
        self.weight2 = orthogonal_parameter(self.HIDDEN_DIM, out_dim, seed + 2)
        self.bias2 = nn.Parameter(torch.zeros(out_dim))

    @property
    def latent_vector(self) -> Tensor:
        return self.fixed_z

    def _hidden(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        raw = torch.matmul(latent, self.weight1) + self.bias1
        return raw, activate(raw, self.activation_name)

    def forward(self, latent_override: Optional[Tensor] = None) -> Tensor:
        latent = self.fixed_z if latent_override is None else latent_override
        _, hidden = self._hidden(latent)
        return torch.matmul(hidden, self.weight2) + self.bias2

    def _effective_first_weight(self) -> Tensor:
        raw, _ = self._hidden(self.fixed_z)
        derivative = activation_derivative(raw, self.activation_name)
        return self.weight1 * derivative.unsqueeze(0)

    def smoothness_loss(self) -> Tensor:
        first = self._effective_first_weight()
        first_gram = first.T @ first
        second_gram = self.weight2 @ self.weight2.T
        return torch.sum(first_gram * second_gram.T)

    def alignment_loss(self) -> Tensor:
        effective = self._effective_first_weight()
        output_row_mean = self.weight2.mean(dim=1)
        latent_row_mean = effective @ output_row_mean
        return 1.0 - F.cosine_similarity(
            self.fixed_z.unsqueeze(0),
            latent_row_mean.unsqueeze(0),
            dim=1,
        ).mean()


class GeneratedCNN2(nn.Module):
    def __init__(self, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.specs = get_param_specs("cnn2")
        self.target_param_count = sum(
            numel_from_shape(shape) for _, shape in self.specs
        )
        if self.target_param_count != TARGET_PARAM_COUNT:
            raise RuntimeError("CNN2 parameter count no longer matches paper Table 7")

    def generate_params(self, perturb_scale: float = 0.0) -> Dict[str, Tensor]:
        latent = self.projector.latent_vector
        if perturb_scale:
            latent = latent + torch.randn_like(latent) * perturb_scale
        flat = self.projector(latent)
        params: Dict[str, Tensor] = {}
        offset = 0
        for name, shape in self.specs:
            size = numel_from_shape(shape)
            params[name] = flat[offset:offset + size].view(shape)
            offset += size
        return params

    def forward(self, x: Tensor) -> Tensor:
        return functional_cnn_forward(x, "cnn2", self.generate_params())

    def smoothness_loss(self) -> Tensor:
        return self.projector.smoothness_loss()

    def alignment_loss(self) -> Tensor:
        return self.projector.alignment_loss()


def validate_protocol(args: argparse.Namespace, spec: VariantSpec) -> None:
    if spec.variant == "baseline" and args.loss_mode != "task":
        raise ValueError("CNN2 baseline uses task cross-entropy only")

    if args.variant == "full-dnn":
        if args.protocol == "paper":
            raise ValueError(
                "The paper does not disclose the Full DNN architecture. "
                "A strict paper run is impossible; use --protocol custom "
                "--allow-undisclosed-full-dnn to run the labeled assumption."
            )
        if not args.allow_undisclosed_full_dnn:
            raise ValueError(
                "Full DNN requires --allow-undisclosed-full-dnn because its "
                "307->61->108618 architecture is only a parameter-count-matching assumption."
            )
    elif args.allow_undisclosed_full_dnn:
        raise ValueError("--allow-undisclosed-full-dnn is only valid for full-dnn")

    if args.protocol != "paper":
        return
    required = {
        "projection_init": "orthogonal",
        "projection_layout": "global",
        "modulation_reduction": "sum",
    }
    for name, expected in required.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"Paper protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isclose(args.weight_scale, 1.0):
        raise ValueError("Paper protocol requires --weight-scale 1")
    if spec.variant != "baseline":
        if args.loss_mode != "full":
            raise ValueError("Paper Table 7 mapping variants use the full Mapping Loss")
        if args.loss_coefficient_mode != "trainable":
            raise ValueError("Paper Mapping Loss coefficients are trainable")
        if set(args.mapping_loss_terms) != set(MAPPING_LOSS_TERMS):
            raise ValueError("Paper protocol requires all three Mapping Loss terms")


def build_model(
    args: argparse.Namespace,
    spec: VariantSpec,
) -> nn.Module:
    if spec.variant == "baseline":
        from train_paper_3_1_1 import CNN2

        model: nn.Module = CNN2()
    elif spec.variant in {"ours", "ours-no-wm"}:
        assert spec.latent_dim is not None
        projector = PaperFixedProjector(
            latent_dim=spec.latent_dim,
            out_dim=TARGET_PARAM_COUNT,
            seed=args.seed,
            chunk_size=args.chunk_size,
            activation_name=args.activation,
            weight_scale=args.weight_scale,
            modulation_scale=(
                0.0 if spec.variant == "ours-no-wm" else args.modulation_scale
            ),
            latent_init_std=args.latent_init_std,
            projection_init=args.projection_init,
            modulation_reduction=args.modulation_reduction,
            projection_layout=args.projection_layout,
        )
        model = GeneratedCNN2(projector)
    elif spec.variant == "lv-wmap":
        assert spec.latent_dim is not None
        projector = SeparateModulationProjector(
            latent_dim=spec.latent_dim,
            out_dim=TARGET_PARAM_COUNT,
            seed=args.seed,
            chunk_size=args.chunk_size,
            activation_name=args.activation,
            weight_scale=args.weight_scale,
            modulation_scale=args.modulation_scale,
            latent_init_std=args.latent_init_std,
            projection_init=args.projection_init,
            modulation_reduction=args.modulation_reduction,
            projection_layout=args.projection_layout,
        )
        model = GeneratedCNN2(projector)
    elif spec.variant == "lv-full-dnn":
        assert spec.latent_dim is not None
        projector = TrainableLinearProjector(
            latent_dim=spec.latent_dim,
            out_dim=TARGET_PARAM_COUNT,
            seed=args.seed,
            activation_name=args.activation,
            weight_scale=args.weight_scale,
            latent_init_std=args.latent_init_std,
            projection_init=args.projection_init,
        )
        model = GeneratedCNN2(projector)
    else:
        projector = AssumedFullDNNProjector(
            out_dim=TARGET_PARAM_COUNT,
            seed=args.seed,
            activation_name=args.activation,
            latent_init_std=args.latent_init_std,
        )
        model = GeneratedCNN2(projector)

    actual = count_trainable_params(model)
    if actual != spec.table_params:
        raise RuntimeError(
            f"{spec.variant} trainable parameters={actual:,}, "
            f"paper Table 7 reports {spec.table_params:,}"
        )
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y).item() * y.numel()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += y.numel()
    if total_count == 0:
        raise RuntimeError("Evaluation consumed no samples")
    return total_loss / total_count, total_correct / total_count


def mapping_batch_loss(
    args: argparse.Namespace,
    model: GeneratedCNN2,
    coefficients: Optional[TrainableMappingLossCoefficients],
    x: Tensor,
    y: Tensor,
) -> Tuple[Tensor, Dict[str, float], Tensor]:
    params = model.generate_params()
    logits = functional_cnn_forward(x, "cnn2", params)
    task = F.cross_entropy(logits, y)
    zero = task.new_zeros(())
    if args.loss_mode == "task":
        return task, {
            "task": float(task.detach()),
            "stability": 0.0,
            "smoothness": 0.0,
            "alignment": 0.0,
        }, logits
    if coefficients is None:
        raise RuntimeError("Full Mapping Loss requires coefficients")

    selected = set(args.mapping_loss_terms)
    if "stability" in selected:
        perturbed = model.generate_params(args.stability_sigma)
        perturbed_logits = functional_cnn_forward(x, "cnn2", perturbed)
        stability = torch.mean(torch.sum((perturbed_logits - logits) ** 2, dim=1))
    else:
        stability = zero
    smoothness = model.smoothness_loss() if "smoothness" in selected else zero
    alignment = model.alignment_loss() if "alignment" in selected else zero
    lambdas = coefficients()
    loss = (
        task
        + lambdas["stability"] * stability
        + lambdas["smoothness"] * smoothness
        + lambdas["alignment"] * alignment
    )
    return loss, {
        "task": float(task.detach()),
        "stability": float(stability.detach()),
        "smoothness": float(smoothness.detach()),
        "alignment": float(alignment.detach()),
    }, logits


def run(args: argparse.Namespace) -> Path:
    spec = resolve_variant_spec(args.variant, args.table_params, args.dataset)
    if args.loss_mode is None:
        args.loss_mode = "task" if spec.variant == "baseline" else "full"
    validate_protocol(args, spec)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_loader, test_loader = get_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_model(args, spec).to(device)

    coefficients: Optional[TrainableMappingLossCoefficients] = None
    optimized_parameters = list(model.parameters())
    if spec.variant != "baseline" and args.loss_mode == "full":
        args.mapping_loss_terms = list(
            validate_mapping_loss_terms(args.mapping_loss_terms)
        )
        coefficients = TrainableMappingLossCoefficients(
            args.lambda_stability,
            args.lambda_smoothness,
            args.lambda_alignment,
            enabled_terms=args.mapping_loss_terms,
        ).to(device)
        optimized_parameters.extend(coefficients.parameters())

    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    run_name = (
        f"{now_string()}_table7_{spec.variant}_{args.dataset}"
        f"_p{spec.table_params}_seed{args.seed}"
    )
    run_dir = ensure_dir(Path(args.results_dir) / run_name)
    coefficient_count = 0 if coefficients is None else count_trainable_params(coefficients)
    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "device_resolved": str(device),
            "variant_spec": asdict(spec),
            "target_param_count": TARGET_PARAM_COUNT,
            "trainable_model_params": count_trainable_params(model),
            "trainable_loss_coefficients": coefficient_count,
            "total_optimized_params": count_trainable_params(model) + coefficient_count,
            "optimizer_name": "AdamW",
            "protocol_scope": (
                "strict_for_disclosed_settings"
                if args.protocol == "paper"
                else "custom_with_explicit_assumptions"
            ),
            "paper_undisclosed_settings": list(PAPER_UNDISCLOSED_SETTINGS),
            "modulation_scale_resolved": (
                0.0 if spec.variant == "ours-no-wm" else args.modulation_scale
            ),
            "full_dnn_assumption": (
                "fixed z 307 -> trainable hidden 61 -> 108618; one of multiple "
                "architectures matching the reported count"
                if spec.variant == "full-dnn"
                else None
            ),
        }
    )
    save_json(config, run_dir / "config.json")

    initial_loss, initial_acc = evaluate(
        model, test_loader, device, args.max_test_batches
    )
    print("=" * 80)
    print("Paper 3.4 Table 7 robustness reproduction")
    print("=" * 80)
    print(f"run_dir                  = {run_dir}")
    print(f"device                   = {device}")
    print(f"variant                  = {spec.variant}")
    print(f"dataset                  = {args.dataset}")
    print(f"table_params             = {spec.table_params:,}")
    print(f"target_generated_params  = {TARGET_PARAM_COUNT:,}")
    print(f"trainable_loss_coeffs    = {coefficient_count}")
    print(f"paper_acc                = {spec.paper_acc * 100:.2f}%")
    print(f"initial_test             = loss {initial_loss:.4f}, acc {initial_acc * 100:.2f}%")
    print(f"implementation           = {spec.implementation_note}")
    print("=" * 80)

    best_acc = -1.0
    best_epoch = 0
    history_path = run_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_task = 0.0
        total_correct = 0
        total_count = 0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        )
        started = time.time()
        for batch_index, (x, y) in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if spec.variant == "baseline":
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                parts = {"task": float(loss.detach())}
            else:
                loss, parts, logits = mapping_batch_loss(
                    args, model, coefficients, x, y
                )
            loss.backward()
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(optimized_parameters, args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_task += parts["task"] * y.numel()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += y.numel()
            progress.set_postfix(
                loss=total_loss / total_count,
                acc=total_correct / total_count,
            )
        if total_count == 0:
            raise RuntimeError("Training consumed no samples")

        test_loss, test_acc = evaluate(
            model, test_loader, device, args.max_test_batches
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
            }
            if coefficients is not None:
                checkpoint["loss_coefficients"] = coefficients.state_dict()
            torch.save(checkpoint, run_dir / "best_model.pt")

        coefficient_values = None if coefficients is None else coefficients.detached_values()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "train_task_loss": total_task / total_count,
            "train_acc": total_correct / total_count,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "paper_acc": spec.paper_acc,
            "mapping_loss_coefficients": coefficient_values,
            "seconds": time.time() - started,
        }
        append_jsonl(row, history_path)
        print(
            f"epoch={epoch:03d} train_acc={row['train_acc'] * 100:.2f}% "
            f"test_acc={test_acc * 100:.2f}% best={best_acc * 100:.2f}% "
            f"paper={spec.paper_acc * 100:.2f}% seconds={row['seconds']:.1f}"
        )

    final = {
        "run_name": run_name,
        "variant": spec.variant,
        "dataset": args.dataset,
        "table_params": spec.table_params,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "paper_acc": spec.paper_acc,
        "gap_best_minus_paper": best_acc - spec.paper_acc,
        "trainable_model_params": count_trainable_params(model),
        "trainable_loss_coefficients": coefficient_count,
        "total_optimized_params": count_trainable_params(model) + coefficient_count,
        "target_param_count": TARGET_PARAM_COUNT,
        "final_mapping_loss_coefficients": (
            None if coefficients is None else coefficients.detached_values()
        ),
        "implementation_note": spec.implementation_note,
    }
    save_json(final, run_dir / "final_result.json")
    print("=" * 80)
    print(f"best_epoch       = {best_epoch}")
    print(f"best_test_acc    = {best_acc * 100:.2f}%")
    print(f"paper_acc        = {spec.paper_acc * 100:.2f}%")
    print(f"saved to         = {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Mapping Networks paper section 3.4, Table 7"
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--table-params",
        type=int,
        help="The # Params value printed in paper Table 7",
    )
    parser.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    parser.add_argument("--protocol", choices=["paper", "custom"], default="paper")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/table7")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)

    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--projection-init",
        choices=["orthogonal", "gaussian"],
        default="orthogonal",
    )
    parser.add_argument(
        "--projection-layout",
        choices=["global", "blockwise"],
        default="global",
    )
    parser.add_argument(
        "--modulation-reduction",
        choices=["sum", "mean"],
        default="sum",
    )
    parser.add_argument(
        "--activation",
        choices=["identity", "tanh", "sin"],
        default="identity",
    )
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--modulation-scale", type=float, default=0.01)
    parser.add_argument("--latent-init-std", type=float, default=0.02)
    parser.add_argument("--stability-sigma", type=float, default=0.01)
    parser.add_argument("--loss-mode", choices=["task", "full"])
    parser.add_argument(
        "--loss-coefficient-mode",
        choices=["trainable"],
        default="trainable",
    )
    parser.add_argument(
        "--mapping-loss-terms",
        nargs="+",
        choices=list(MAPPING_LOSS_TERMS),
        default=list(MAPPING_LOSS_TERMS),
    )
    parser.add_argument("--lambda-stability", type=float, default=0.05)
    parser.add_argument("--lambda-smoothness", type=float, default=5e-6)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    parser.add_argument("--allow-undisclosed-full-dnn", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
