from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from mapping_networks.models import (
    DirectCNN,
    DirectCNN1,
    DirectCNN2,
    DirectMLP,
    EfficientCNN,
    MappingMLP,
    ProjectionMappingCNN,
    ProjectionMappingCNN1,
    ProjectionMappingCNN2,
    ProjectionMappingEfficientCNN,
    ProjectionMappingMLP,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mapping Network toy classifier.")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument(
        "--model",
        choices=[
            "direct",
            "direct-cnn",
            "cnn1",
            "cnn2",
            "efficient-cnn",
            "mapping",
            "projection",
            "projection-cnn",
            "projection-cnn1",
            "projection-cnn2",
            "projection-efficient-cnn",
        ],
        default="projection-efficient-cnn",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--mapper-width", type=int, default=128)
    parser.add_argument("--mapper-depth", type=int, default=3)
    parser.add_argument("--train-mapper", action="store_true")
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument(
        "--layerwise-latent-dims",
        type=int,
        nargs="+",
        default=None,
        help="Optional per-layer latent dimensions for projection-cnn1/projection-cnn2.",
    )
    parser.add_argument("--modulation-scale", type=float, default=0.01)
    parser.add_argument("--activation", choices=["tanh", "identity"], default="tanh")
    parser.add_argument("--projection-gain", type=float, default=1.0)
    parser.add_argument("--latent-init-std", type=float, default=1.0)
    parser.add_argument("--output-gain", type=float, default=1.0)
    parser.add_argument(
        "--parameter-scale-mode",
        choices=["fan-in", "paper"],
        default="fan-in",
        help="Use paper to keep generated parameters as sigma(Wz+b); fan-in keeps the older stabilizing scale.",
    )
    parser.add_argument("--mapping-loss-weight", type=float, default=0.0)
    parser.add_argument("--mapping-stability-weight", type=float, default=0.05)
    parser.add_argument("--mapping-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--mapping-alignment-weight", type=float, default=0.05)
    parser.add_argument("--mapping-perturb-std", type=float, default=1e-3)
    parser.add_argument(
        "--fixed-mapping-loss-coefficients",
        action="store_true",
        help="Use fixed mapping-loss coefficients instead of the paper's trainable lambdas.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a tiny split for smoke tests.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    train_set = dataset_cls(args.data_dir, train=True, transform=transform, download=True)
    test_set = dataset_cls(args.data_dir, train=False, transform=transform, download=True)

    if args.quick:
        train_set = Subset(train_set, range(2048))
        test_set = Subset(test_set, range(1024))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "direct":
        return DirectMLP(hidden_dims=args.hidden_dims)
    if args.model == "direct-cnn":
        return DirectCNN()
    if args.model == "cnn1":
        return DirectCNN1()
    if args.model == "cnn2":
        return DirectCNN2()
    if args.model == "efficient-cnn":
        return EfficientCNN()
    if args.model == "projection-cnn":
        return ProjectionMappingCNN(
            latent_dim=args.latent_dim,
            modulation_scale=args.modulation_scale,
            activation=args.activation,
            layerwise=args.layerwise,
            projection_gain=args.projection_gain,
            latent_init_std=args.latent_init_std,
            output_gain=args.output_gain,
            parameter_scale_mode=args.parameter_scale_mode,
        )
    if args.model == "projection-cnn1":
        return ProjectionMappingCNN1(
            latent_dim=args.latent_dim,
            modulation_scale=args.modulation_scale,
            activation=args.activation,
            layerwise=args.layerwise,
            layerwise_latent_dims=args.layerwise_latent_dims,
            projection_gain=args.projection_gain,
            latent_init_std=args.latent_init_std,
            output_gain=args.output_gain,
            parameter_scale_mode=args.parameter_scale_mode,
        )
    if args.model == "projection-cnn2":
        return ProjectionMappingCNN2(
            latent_dim=args.latent_dim,
            modulation_scale=args.modulation_scale,
            activation=args.activation,
            layerwise=args.layerwise,
            layerwise_latent_dims=args.layerwise_latent_dims,
            projection_gain=args.projection_gain,
            latent_init_std=args.latent_init_std,
            output_gain=args.output_gain,
            parameter_scale_mode=args.parameter_scale_mode,
        )
    if args.model == "projection-efficient-cnn":
        return ProjectionMappingEfficientCNN(
            latent_dim=args.latent_dim,
            modulation_scale=args.modulation_scale,
            activation=args.activation,
            layerwise=args.layerwise,
            projection_gain=args.projection_gain,
            latent_init_std=args.latent_init_std,
            output_gain=args.output_gain,
        )
    if args.model == "projection":
        return ProjectionMappingMLP(
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            modulation_scale=args.modulation_scale,
            activation=args.activation,
            layerwise=args.layerwise,
            projection_gain=args.projection_gain,
            latent_init_std=args.latent_init_std,
            output_gain=args.output_gain,
        )
    return MappingMLP(
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        mapper_width=args.mapper_width,
        mapper_depth=args.mapper_depth,
        train_mapper=args.train_mapper,
    )


class MappingLossCoefficients(nn.Module):
    """Trainable positive lambdas for the paper's Mapping Loss."""

    def __init__(self, stability: float = 0.05, smoothness: float = 0.05, alignment: float = 0.05) -> None:
        super().__init__()
        initial = torch.tensor([stability, smoothness, alignment], dtype=torch.float32)
        if torch.any(initial <= 0):
            raise ValueError("Mapping-loss coefficient initial values must be positive.")
        self.raw = nn.Parameter(torch.log(torch.expm1(initial)))

    def forward(self) -> dict[str, torch.Tensor]:
        values = torch.nn.functional.softplus(self.raw)
        return {
            "stability": values[0],
            "smoothness": values[1],
            "alignment": values[2],
        }


def compute_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    *,
    mapping_loss_weight: float = 0.0,
    mapping_stability_weight: float = 1.0,
    mapping_smoothness_weight: float = 1.0,
    mapping_alignment_weight: float = 1.0,
    mapping_loss_coefficients: MappingLossCoefficients | None = None,
    mapping_perturb_std: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    logits = model(images)
    task_loss = criterion(logits, labels)

    if mapping_loss_weight <= 0:
        return task_loss, {"task": task_loss.detach(), "mapping": 0.0}

    if not hasattr(model, "mapping_loss_terms"):
        raise ValueError("--mapping-loss-weight requires a projection model with mapping_loss_terms().")

    mapping_terms = model.mapping_loss_terms(
        images,
        clean_logits=logits,
        perturb_std=mapping_perturb_std,
    )
    if mapping_loss_coefficients is None:
        coefficients = {
            "stability": torch.as_tensor(mapping_stability_weight, device=images.device),
            "smoothness": torch.as_tensor(mapping_smoothness_weight, device=images.device),
            "alignment": torch.as_tensor(mapping_alignment_weight, device=images.device),
        }
    else:
        coefficients = mapping_loss_coefficients()

    mapping_loss = (
        coefficients["stability"] * mapping_terms["stability"]
        + coefficients["smoothness"] * mapping_terms["smoothness"]
        + coefficients["alignment"] * mapping_terms["alignment"]
    )
    total_loss = task_loss + mapping_loss_weight * mapping_loss

    terms: dict[str, torch.Tensor | float] = {
        "task": task_loss.detach(),
        "mapping": mapping_loss.detach(),
        "mapping_stability": mapping_terms["stability"].detach(),
        "mapping_smoothness": mapping_terms["smoothness"].detach(),
        "mapping_alignment": mapping_terms["alignment"].detach(),
        "lambda_stability": coefficients["stability"].detach(),
        "lambda_smoothness": coefficients["smoothness"].detach(),
        "lambda_alignment": coefficients["alignment"].detach(),
    }
    return total_loss, terms


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    show_progress: bool,
    mapping_loss_weight: float = 0.0,
    mapping_stability_weight: float = 1.0,
    mapping_smoothness_weight: float = 1.0,
    mapping_alignment_weight: float = 1.0,
    mapping_loss_coefficients: MappingLossCoefficients | None = None,
    mapping_perturb_std: float = 1e-3,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for images, labels in tqdm(loader, leave=False, disable=not show_progress):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_loss(
            model,
            images,
            labels,
            criterion,
            mapping_loss_weight=mapping_loss_weight,
            mapping_stability_weight=mapping_stability_weight,
            mapping_smoothness_weight=mapping_smoothness_weight,
            mapping_alignment_weight=mapping_alignment_weight,
            mapping_loss_coefficients=mapping_loss_coefficients,
            mapping_perturb_std=mapping_perturb_std,
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total += labels.numel()
    return total_loss / total


def warn_if_tanh_mapping_is_likely_saturated(model: nn.Module, args: argparse.Namespace) -> None:
    if args.activation != "tanh" or not hasattr(model, "_latent_projection_biases"):
        return

    with torch.no_grad():
        shifts = [
            float(args.modulation_scale * torch.sum(latent.detach() * latent.detach()).item())
            for _, _, latent in model._latent_projection_biases()
        ]

    max_abs_shift = max((abs(shift) for shift in shifts), default=0.0)
    if max_abs_shift < 3.0:
        return

    print(
        "warning=tanh_mapping_saturation_risk "
        f"max_modulation_shift={max_abs_shift:.4f} "
        "tanh is nearly saturated above about 3; "
        "try a smaller --latent-init-std or --modulation-scale."
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    train_loader, test_loader = build_loaders(args)
    model = build_model(args).to(device)
    warn_if_tanh_mapping_is_likely_saturated(model, args)
    mapping_loss_coefficients = None
    if args.mapping_loss_weight > 0 and not args.fixed_mapping_loss_coefficients:
        mapping_loss_coefficients = MappingLossCoefficients(
            args.mapping_stability_weight,
            args.mapping_smoothness_weight,
            args.mapping_alignment_weight,
        ).to(device)

    optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
    if mapping_loss_coefficients is not None:
        optimizer_parameters.extend(mapping_loss_coefficients.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    mapping_loss_trainable = (
        sum(parameter.numel() for parameter in mapping_loss_coefficients.parameters())
        if mapping_loss_coefficients is not None
        else 0
    )
    target_count = getattr(model, "target_parameter_count", count_parameters(model))
    print(
        f"device={device} model={args.model} trainable={trainable:,} "
        f"target_params={target_count:,} mapping_loss_trainable={mapping_loss_trainable:,}"
    )

    initial_test_loss, initial_test_acc = evaluate(model, test_loader, device)
    print(f"initial test_loss={initial_test_loss:.4f} test_acc={initial_test_acc:.4f}")

    history = []
    started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            show_progress=not args.no_progress,
            mapping_loss_weight=args.mapping_loss_weight,
            mapping_stability_weight=args.mapping_stability_weight,
            mapping_smoothness_weight=args.mapping_smoothness_weight,
            mapping_alignment_weight=args.mapping_alignment_weight,
            mapping_loss_coefficients=mapping_loss_coefficients,
            mapping_perturb_std=args.mapping_perturb_std,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        epoch_seconds = time.perf_counter() - epoch_started_at
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} seconds={epoch_seconds:.1f}"
        )

    elapsed_seconds = time.perf_counter() - started_at
    result = {
        "args": vars(args) | {"data_dir": str(args.data_dir), "output": str(args.output)},
        "device": str(device),
        "trainable_parameters": trainable,
        "mapping_loss_trainable_parameters": mapping_loss_trainable,
        "target_parameters": target_count,
        "initial_test_loss": initial_test_loss,
        "initial_test_acc": initial_test_acc,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_epoch": elapsed_seconds / max(1, args.epochs),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
