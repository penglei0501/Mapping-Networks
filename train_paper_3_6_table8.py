from __future__ import annotations

import argparse
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm

from train_paper_3_1_1 import (
    CNN2,
    ChunkedFixedProjector,
    TrainableMappingLossCoefficients,
    append_jsonl,
    count_trainable_params,
    ensure_dir,
    get_loaders,
    now_string,
    numel_from_shape,
    resolve_device,
    save_json,
    seed_everything,
    validate_layerwise_dims,
    validate_layerwise_modulation_scales,
)


PAPER_TABLE_8 = {
    ("cnn2-lrd", "mnist"): (35_914, 0.9812),
    ("cnn2-lrd", "fmnist"): (35_914, 0.8967),
    ("cnn2-prune", "mnist"): (10_862, 0.9587),
    ("cnn2-prune", "fmnist"): (10_862, 0.8791),
    ("slvt-lrd", "mnist"): (1_456, 0.9780),
    ("slvt-lrd", "fmnist"): (1_456, 0.9067),
    ("slvt-prune", "mnist"): (2_048, 0.9593),
    ("slvt-prune", "fmnist"): (2_048, 0.8870),
    ("lwt-lrd", "mnist"): (2_688, 0.9881),
    ("lwt-lrd", "fmnist"): (2_688, 0.9355),
    ("lwt-prune", "mnist"): (2_688, 0.9715),
    ("lwt-prune", "fmnist"): (2_688, 0.9179),
}

TARGET_DENSE_PARAMS = 108_618
TABLE_LRD_PARAMS = 35_914
TABLE_PRUNED_PARAMS = 10_862
TABLE_COUNT_MATCHING_RANK = 32
PAPER_PROSE_RANK = 16
MAPPING_LOSS_TERMS = ("stability", "smoothness", "alignment")

VARIANTS = tuple(sorted({variant for variant, _ in PAPER_TABLE_8}))


def dense_specs() -> List[Tuple[str, Tuple[int, ...]]]:
    return [
        ("conv1.weight", (16, 1, 3, 3)),
        ("conv1.bias", (16,)),
        ("conv2.weight", (32, 16, 3, 3)),
        ("conv2.bias", (32,)),
        ("fc1.v", (128, 800)),
        ("fc1.bias", (128,)),
        ("fc2.weight", (10, 128)),
        ("fc2.bias", (10,)),
    ]


def low_rank_specs(rank: int) -> List[Tuple[str, Tuple[int, ...]]]:
    return [
        ("conv1.weight", (16, 1, 3, 3)),
        ("conv1.bias", (16,)),
        ("conv2.weight", (32, 16, 3, 3)),
        ("conv2.bias", (32,)),
        ("fc1.v", (rank, 800)),
        ("fc1.u", (128, rank)),
        ("fc1.bias", (128,)),
        ("fc2.weight", (10, 128)),
        ("fc2.bias", (10,)),
    ]


def count_specs(specs: Iterable[Tuple[str, Tuple[int, ...]]]) -> int:
    return sum(numel_from_shape(shape) for _, shape in specs)


class LowRankCNN2(nn.Module):
    """CNN2 with only FC1 factorized as U @ V."""

    def __init__(self, rank: int = TABLE_COUNT_MATCHING_RANK) -> None:
        super().__init__()
        self.rank = int(rank)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1_v = nn.Linear(800, self.rank, bias=False)
        self.fc1_u = nn.Linear(self.rank, 128, bias=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1_u(self.fc1_v(x)))
        return self.fc2(x)


def functional_addon_forward(
    x: Tensor,
    params: Mapping[str, Tensor],
    low_rank: bool,
) -> Tensor:
    x = F.conv2d(x, params["conv1.weight"], params["conv1.bias"])
    x = F.max_pool2d(F.relu(x), 2)
    x = F.conv2d(x, params["conv2.weight"], params["conv2.bias"])
    x = F.max_pool2d(F.relu(x), 2)
    x = torch.flatten(x, 1)
    if low_rank:
        x = F.linear(x, params["fc1.v"], None)
        x = F.linear(x, params["fc1.u"], params["fc1.bias"])
    else:
        x = F.linear(x, params["fc1.v"], params["fc1.bias"])
    x = F.relu(x)
    return F.linear(x, params["fc2.weight"], params["fc2.bias"])


def exact_global_magnitude_masks(
    tensors: Mapping[str, Tensor],
    keep_count: int,
) -> Dict[str, Tensor]:
    total = sum(tensor.numel() for tensor in tensors.values())
    if keep_count <= 0 or keep_count > total:
        raise ValueError(f"keep_count must be in [1, {total}], got {keep_count}")
    flat = torch.cat([tensor.detach().abs().reshape(-1) for tensor in tensors.values()])
    kept_indices = torch.topk(flat, k=keep_count, largest=True, sorted=False).indices
    flat_mask = torch.zeros(total, dtype=torch.bool, device=flat.device)
    flat_mask[kept_indices] = True
    masks: Dict[str, Tensor] = OrderedDict()
    offset = 0
    for name, tensor in tensors.items():
        end = offset + tensor.numel()
        masks[name] = flat_mask[offset:end].view_as(tensor)
        offset = end
    return masks


def exact_tensorwise_magnitude_masks(
    tensors: Mapping[str, Tensor],
    keep_fraction: float,
) -> Dict[str, Tensor]:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    masks: Dict[str, Tensor] = OrderedDict()
    for name, tensor in tensors.items():
        keep_count = max(1, round(tensor.numel() * keep_fraction))
        flat = tensor.detach().abs().reshape(-1)
        kept_indices = torch.topk(
            flat, k=keep_count, largest=True, sorted=False
        ).indices
        mask = torch.zeros(
            tensor.numel(), dtype=torch.bool, device=tensor.device
        )
        mask[kept_indices] = True
        masks[name] = mask.view_as(tensor)
    return masks


def masked_params(
    params: Mapping[str, Tensor],
    keep_count: int,
    scope: str = "global",
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    if scope == "global":
        masks = exact_global_magnitude_masks(params, keep_count)
    elif scope == "tensor":
        total = sum(value.numel() for value in params.values())
        masks = exact_tensorwise_magnitude_masks(
            params, keep_fraction=keep_count / total
        )
        actual = sum(int(mask.sum()) for mask in masks.values())
        if actual != keep_count:
            raise RuntimeError(
                f"Tensor-wise pruning kept {actual:,}, expected {keep_count:,}"
            )
    else:
        raise ValueError(f"Unknown pruning scope: {scope}")
    masked = OrderedDict((name, value * masks[name]) for name, value in params.items())
    return masked, masks


def apply_tensor_masks(
    params: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    if params.keys() != masks.keys():
        raise ValueError("Parameter and pruning-mask keys must match")
    return OrderedDict(
        (name, value * masks[name]) for name, value in params.items()
    )


def apply_global_pruning(model: nn.Module, keep_count: int) -> Dict[str, Tensor]:
    named = OrderedDict(model.named_parameters())
    masks = exact_global_magnitude_masks(named, keep_count)
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.mul_(masks[name])
    return masks


def enforce_parameter_masks(
    model: nn.Module,
    masks: Mapping[str, Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.mul_(masks[name])


def register_parameter_mask_hooks(
    model: nn.Module,
    masks: Mapping[str, Tensor],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles = []
    for name, parameter in model.named_parameters():
        mask = masks[name]
        handles.append(
            parameter.register_hook(
                lambda gradient, fixed_mask=mask: gradient * fixed_mask
            )
        )
    return handles


def count_nonzero_tensors(tensors: Mapping[str, Tensor]) -> int:
    return sum(int(torch.count_nonzero(tensor)) for tensor in tensors.values())


class AddonMappingCNN(nn.Module):
    def __init__(
        self,
        mode: str,
        low_rank: bool,
        latent_dim: Optional[int],
        total_latent_dim: Optional[int],
        layerwise_latent_dims: Optional[List[int]],
        seed: int,
        chunk_size: int,
        activation: str,
        weight_scale: float,
        modulation_scale: float,
        layerwise_modulation_scales: Optional[List[float]],
        latent_init_std: float,
        projection_init: str,
        modulation_reduction: str,
        projection_layout: str,
        lrd_rank: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.low_rank = bool(low_rank)
        self.specs = low_rank_specs(lrd_rank) if low_rank else dense_specs()
        self.target_param_count = count_specs(self.specs)
        self.layer_groups = self._build_layer_groups()

        common = {
            "chunk_size": chunk_size,
            "activation": activation,
            "weight_scale": weight_scale,
            "modulation_scale": modulation_scale,
            "latent_init_std": latent_init_std,
            "projection_init": projection_init,
            "modulation_reduction": modulation_reduction,
            "projection_layout": projection_layout,
        }
        if mode == "slvt":
            if latent_dim is None:
                raise ValueError("SLVT requires latent_dim")
            self.projector = ChunkedFixedProjector(
                latent_dim=latent_dim,
                out_dim=self.target_param_count,
                seed=seed,
                **common,
            )
            self.layer_dims = None
            self.layer_modulation_scales = None
        elif mode == "lwt":
            if total_latent_dim is None:
                raise ValueError("LWT requires total_latent_dim")
            group_specs = [
                (
                    group,
                    (sum(numel_from_shape(shape) for _, shape in items),),
                )
                for group, items in self.layer_groups.items()
            ]
            if layerwise_latent_dims is None:
                raise ValueError(
                    "Table 8 LWT requires an explicit auditable latent allocation"
                )
            self.layer_dims = validate_layerwise_dims(
                group_specs, total_latent_dim, layerwise_latent_dims
            )
            self.layer_modulation_scales = validate_layerwise_modulation_scales(
                self.layer_groups.keys(),
                modulation_scale,
                layerwise_modulation_scales,
            )
            self.projectors = nn.ModuleDict()
            for index, (group, items) in enumerate(self.layer_groups.items()):
                group_out = sum(numel_from_shape(shape) for _, shape in items)
                group_common = dict(common)
                group_common["modulation_scale"] = self.layer_modulation_scales[group]
                self.projectors[group] = ChunkedFixedProjector(
                    latent_dim=self.layer_dims[group],
                    out_dim=group_out,
                    seed=seed + 100000 + index * 997,
                    **group_common,
                )
        else:
            raise ValueError("mode must be slvt or lwt")

    def _build_layer_groups(self) -> Dict[str, List[Tuple[str, Tuple[int, ...]]]]:
        groups: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = OrderedDict()
        for name, shape in self.specs:
            groups.setdefault(name.split(".")[0], []).append((name, shape))
        return groups

    def trainable_latent_count(self) -> int:
        if self.mode == "slvt":
            return self.projector.z.numel()
        return sum(projector.z.numel() for projector in self.projectors.values())

    def generate_params(self, perturb_scale: float = 0.0) -> Dict[str, Tensor]:
        params: Dict[str, Tensor] = OrderedDict()
        if self.mode == "slvt":
            latent = self.projector.z
            flat = (
                self.projector()
                if perturb_scale == 0.0
                else self.projector(latent + torch.randn_like(latent) * perturb_scale)
            )
            offset = 0
            for name, shape in self.specs:
                size = numel_from_shape(shape)
                params[name] = flat[offset : offset + size].view(shape)
                offset += size
            return params

        for group, items in self.layer_groups.items():
            projector = self.projectors[group]
            latent = projector.z
            flat = (
                projector()
                if perturb_scale == 0.0
                else projector(latent + torch.randn_like(latent) * perturb_scale)
            )
            offset = 0
            for name, shape in items:
                size = numel_from_shape(shape)
                params[name] = flat[offset : offset + size].view(shape)
                offset += size
        return params

    def smoothness_loss(self) -> Tensor:
        if self.mode == "slvt":
            return self.projector.smoothness_loss()
        return torch.stack(
            [projector.smoothness_loss() for projector in self.projectors.values()]
        ).sum()

    def alignment_loss(self) -> Tensor:
        if self.mode == "slvt":
            return self.projector.alignment_loss()
        return torch.stack(
            [projector.alignment_loss() for projector in self.projectors.values()]
        ).mean()

    def forward(self, x: Tensor) -> Tensor:
        return functional_addon_forward(x, self.generate_params(), self.low_rank)


def evaluate_module(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y).item() * y.numel()
            total_correct += int((logits.argmax(1) == y).sum())
            total_count += y.numel()
    if total_count == 0:
        raise RuntimeError("Evaluation consumed no samples")
    return total_loss / total_count, total_correct / total_count


def evaluate_mapping(
    model: AddonMappingCNN,
    loader,
    device: torch.device,
    max_batches: Optional[int],
    prune_keep_count: Optional[int] = None,
    fixed_masks: Optional[Mapping[str, Tensor]] = None,
    pruning_scope: str = "global",
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        params = model.generate_params()
        if fixed_masks is not None:
            params = apply_tensor_masks(params, fixed_masks)
        elif prune_keep_count is not None:
            params, _ = masked_params(
                params, prune_keep_count, scope=pruning_scope
            )
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = functional_addon_forward(x, params, model.low_rank)
            total_loss += F.cross_entropy(logits, y).item() * y.numel()
            total_correct += int((logits.argmax(1) == y).sum())
            total_count += y.numel()
    if total_count == 0:
        raise RuntimeError("Evaluation consumed no samples")
    return total_loss / total_count, total_correct / total_count


def load_state_dict(path: Path, device: torch.device) -> Mapping[str, Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_pruning_masks(
    path: Path,
    device: torch.device,
    params: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    if set(loaded) != set(params):
        raise ValueError("Saved pruning-mask keys do not match generated parameters")
    masks: Dict[str, Tensor] = OrderedDict()
    for name, value in params.items():
        mask = loaded[name].to(device=device, dtype=torch.bool)
        if mask.shape != value.shape:
            raise ValueError(
                f"Mask shape for {name} is {tuple(mask.shape)}, "
                f"expected {tuple(value.shape)}"
            )
        masks[name] = mask
    actual = sum(int(mask.sum()) for mask in masks.values())
    if actual != TABLE_PRUNED_PARAMS:
        raise ValueError(
            f"Saved masks retain {actual:,} values, expected {TABLE_PRUNED_PARAMS:,}"
        )
    return masks


def run_direct(args: argparse.Namespace, run_dir: Path, paper_acc: float) -> dict:
    device = resolve_device(args.device)
    train_loader, test_loader = get_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.seed
    )
    is_lrd = args.variant == "cnn2-lrd"
    model: nn.Module = LowRankCNN2(args.lrd_rank) if is_lrd else CNN2()
    model = model.to(device)
    initial_loss, initial_acc = evaluate_module(
        model, test_loader, device, args.max_test_batches
    )
    print(f"initial_test             = loss {initial_loss:.4f}, acc {initial_acc:.2%}")

    best_acc = -1.0
    best_epoch = 0
    dense_acc: Optional[float] = None
    if args.source_checkpoint:
        model.load_state_dict(load_state_dict(Path(args.source_checkpoint), device))
        dense_loss, dense_acc = evaluate_module(
            model, test_loader, device, args.max_test_batches
        )
        best_acc = dense_acc
        print(f"loaded_dense_test        = loss {dense_loss:.4f}, acc {dense_acc:.2%}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            started = time.time()
            progress = tqdm(
                train_loader,
                desc=f"epoch {epoch}/{args.epochs}",
                leave=False,
                disable=args.no_progress,
            )
            for index, (x, y) in enumerate(progress):
                if args.max_train_batches is not None and index >= args.max_train_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.numel()
                total_correct += int((logits.argmax(1) == y).sum())
                total_count += y.numel()
            test_loss, test_acc = evaluate_module(
                model, test_loader, device, args.max_test_batches
            )
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "best_model.pt")
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_count,
                "train_acc": total_correct / total_count,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_test_acc": best_acc,
                "best_epoch": best_epoch,
                "seconds": time.time() - started,
            }
            append_jsonl(row, run_dir / "history.jsonl")
            print(
                f"epoch={epoch:03d} train_acc={row['train_acc']:.2%} "
                f"test_acc={test_acc:.2%} best={best_acc:.2%}"
            )
        model.load_state_dict(
            load_state_dict(run_dir / "best_model.pt", device)
        )
        dense_acc = best_acc

    effective_nonzero = count_trainable_params(model)
    addon_acc = best_acc
    if args.variant == "cnn2-prune":
        masks = apply_global_pruning(model, TABLE_PRUNED_PARAMS)
        torch.save({name: mask.cpu() for name, mask in masks.items()}, run_dir / "pruning_masks.pt")
        pruned_loss, addon_acc = evaluate_module(
            model, test_loader, device, args.max_test_batches
        )
        initial_pruned_acc = addon_acc
        best_pruned_acc = addon_acc
        best_pruned_epoch = 0
        torch.save(model.state_dict(), run_dir / "best_pruned_model.pt")
        effective_nonzero = count_nonzero_tensors(OrderedDict(model.named_parameters()))
        print(
            f"initial_pruned_test      = loss {pruned_loss:.4f}, "
            f"acc {addon_acc:.2%}, nonzero {effective_nonzero:,}"
        )
        if args.prune_finetune_epochs > 0:
            handles = register_parameter_mask_hooks(model, masks)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.prune_finetune_lr,
                weight_decay=args.weight_decay,
            )
            for epoch in range(1, args.prune_finetune_epochs + 1):
                model.train()
                total_loss = 0.0
                total_correct = 0
                total_count = 0
                started = time.time()
                progress = tqdm(
                    train_loader,
                    desc=f"prune-ft {epoch}/{args.prune_finetune_epochs}",
                    leave=False,
                    disable=args.no_progress,
                )
                for index, (x, y) in enumerate(progress):
                    if args.max_train_batches is not None and index >= args.max_train_batches:
                        break
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()
                    enforce_parameter_masks(model, masks)
                    total_loss += loss.item() * y.numel()
                    total_correct += int((logits.argmax(1) == y).sum())
                    total_count += y.numel()
                test_loss, test_acc = evaluate_module(
                    model, test_loader, device, args.max_test_batches
                )
                if test_acc > best_pruned_acc:
                    best_pruned_acc = test_acc
                    best_pruned_epoch = epoch
                    torch.save(model.state_dict(), run_dir / "best_pruned_model.pt")
                row = {
                    "finetune_epoch": epoch,
                    "train_loss": total_loss / total_count,
                    "train_acc": total_correct / total_count,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best_test_acc": best_pruned_acc,
                    "best_epoch": best_pruned_epoch,
                    "effective_nonzero_target_params": count_nonzero_tensors(
                        OrderedDict(model.named_parameters())
                    ),
                    "seconds": time.time() - started,
                }
                append_jsonl(row, run_dir / "prune_history.jsonl")
                print(
                    f"prune_ft={epoch:03d} train_acc={row['train_acc']:.2%} "
                    f"test_acc={test_acc:.2%} best={best_pruned_acc:.2%}"
                )
            for handle in handles:
                handle.remove()
            model.load_state_dict(
                load_state_dict(run_dir / "best_pruned_model.pt", device)
            )
            enforce_parameter_masks(model, masks)
            addon_acc = best_pruned_acc
            best_epoch = best_pruned_epoch
        else:
            best_epoch = 0

    return {
        "best_epoch": best_epoch,
        "dense_test_acc": dense_acc,
        "best_test_acc": addon_acc,
        "initial_pruned_test_acc": (
            initial_pruned_acc if args.variant == "cnn2-prune" else None
        ),
        "effective_nonzero_target_params": effective_nonzero,
        "trainable_model_params": (
            effective_nonzero
            if args.variant == "cnn2-prune"
            else count_trainable_params(model)
        ),
        "stored_model_params": count_trainable_params(model),
        "trainable_loss_coefficients": 0,
        "final_mapping_loss_coefficients": None,
    }


def mapping_batch_loss(
    args: argparse.Namespace,
    model: AddonMappingCNN,
    coefficients: Optional[TrainableMappingLossCoefficients],
    x: Tensor,
    y: Tensor,
    fixed_masks: Optional[Mapping[str, Tensor]] = None,
) -> Tuple[Tensor, Tensor, float]:
    params = model.generate_params()
    if fixed_masks is not None:
        params = apply_tensor_masks(params, fixed_masks)
    logits = functional_addon_forward(x, params, model.low_rank)
    task = F.cross_entropy(logits, y)
    if args.loss_mode == "task":
        return task, logits, float(task.detach())
    if coefficients is None:
        raise RuntimeError("Full Mapping Loss requires trainable coefficients")
    perturbed = model.generate_params(args.stability_sigma)
    if fixed_masks is not None:
        perturbed = apply_tensor_masks(perturbed, fixed_masks)
    perturbed_logits = functional_addon_forward(x, perturbed, model.low_rank)
    stability = torch.mean(torch.sum((perturbed_logits - logits) ** 2, dim=1))
    smoothness = model.smoothness_loss()
    alignment = model.alignment_loss()
    lambdas = coefficients()
    loss = (
        task
        + lambdas["stability"] * stability
        + lambdas["smoothness"] * smoothness
        + lambdas["alignment"] * alignment
    )
    return loss, logits, float(task.detach())


def make_mapping_model(args: argparse.Namespace) -> AddonMappingCNN:
    mode = "slvt" if args.variant.startswith("slvt") else "lwt"
    low_rank = args.variant.endswith("lrd")
    latent_dim = 1_456 if args.variant == "slvt-lrd" else 2_048
    total_latent_dim = 2_688 if mode == "lwt" else None
    return AddonMappingCNN(
        mode=mode,
        low_rank=low_rank,
        latent_dim=latent_dim if mode == "slvt" else None,
        total_latent_dim=total_latent_dim,
        layerwise_latent_dims=args.layerwise_latent_dims,
        seed=args.seed,
        chunk_size=args.chunk_size,
        activation=args.activation,
        weight_scale=args.weight_scale,
        modulation_scale=args.modulation_scale,
        layerwise_modulation_scales=args.layerwise_modulation_scales,
        latent_init_std=args.latent_init_std,
        projection_init=args.projection_init,
        modulation_reduction=args.modulation_reduction,
        projection_layout=args.projection_layout,
        lrd_rank=args.lrd_rank,
    )


def finetune_pruned_mapping(
    args: argparse.Namespace,
    model: AddonMappingCNN,
    coefficients: Optional[TrainableMappingLossCoefficients],
    optimized: List[nn.Parameter],
    train_loader,
    test_loader,
    device: torch.device,
    fixed_masks: Mapping[str, Tensor],
    run_dir: Path,
    initial_acc: float,
) -> Tuple[float, int]:
    best_acc = initial_acc
    best_epoch = 0
    torch.save(model.state_dict(), run_dir / "best_mapping_model.pt")
    if args.prune_finetune_epochs == 0:
        return best_acc, best_epoch

    optimizer = torch.optim.AdamW(
        optimized,
        lr=args.prune_finetune_lr,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, args.prune_finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        total_task = 0.0
        total_correct = 0
        total_count = 0
        started = time.time()
        progress = tqdm(
            train_loader,
            desc=f"prune-ft {epoch}/{args.prune_finetune_epochs}",
            leave=False,
            disable=args.no_progress,
        )
        for index, (x, y) in enumerate(progress):
            if args.max_train_batches is not None and index >= args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, task = mapping_batch_loss(
                args,
                model,
                coefficients,
                x,
                y,
                fixed_masks=fixed_masks,
            )
            loss.backward()
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(optimized, args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_task += task * y.numel()
            total_correct += int((logits.argmax(1) == y).sum())
            total_count += y.numel()
        test_loss, test_acc = evaluate_mapping(
            model,
            test_loader,
            device,
            args.max_test_batches,
            fixed_masks=fixed_masks,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_mapping_model.pt")
        row = {
            "finetune_epoch": epoch,
            "train_loss": total_loss / total_count,
            "train_task_loss": total_task / total_count,
            "train_acc": total_correct / total_count,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "effective_nonzero_target_params": TABLE_PRUNED_PARAMS,
            "mapping_loss_coefficients": (
                None if coefficients is None else coefficients.detached_values()
            ),
            "seconds": time.time() - started,
        }
        append_jsonl(row, run_dir / "prune_history.jsonl")
        print(
            f"prune_ft={epoch:03d} train_acc={row['train_acc']:.2%} "
            f"test_acc={test_acc:.2%} best={best_acc:.2%}"
        )
    model.load_state_dict(
        load_state_dict(run_dir / "best_mapping_model.pt", device)
    )
    return best_acc, best_epoch


def run_mapping(args: argparse.Namespace, run_dir: Path, paper_acc: float) -> dict:
    device = resolve_device(args.device)
    train_loader, test_loader = get_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.seed
    )
    model = make_mapping_model(args).to(device)
    latent_count = model.trainable_latent_count()
    table_params = PAPER_TABLE_8[(args.variant, args.dataset)][0]
    if latent_count != table_params:
        raise RuntimeError(
            f"Trainable latent count {latent_count:,} != Table 8 {table_params:,}"
        )
    prune_keep = TABLE_PRUNED_PARAMS if args.variant.endswith("prune") else None

    coefficients: Optional[TrainableMappingLossCoefficients] = None
    optimized = list(model.parameters())
    if args.loss_mode == "full":
        coefficients = TrainableMappingLossCoefficients(
            args.lambda_stability,
            args.lambda_smoothness,
            args.lambda_alignment,
            MAPPING_LOSS_TERMS,
        ).to(device)
        optimized.extend(coefficients.parameters())

    best_acc = -1.0
    best_epoch = 0
    dense_acc: Optional[float] = None
    initial_pruned_acc: Optional[float] = None
    fixed_masks: Optional[Dict[str, Tensor]] = None
    if args.source_checkpoint:
        model.load_state_dict(load_state_dict(Path(args.source_checkpoint), device))
        _, dense_acc = evaluate_mapping(
            model, test_loader, device, args.max_test_batches
        )
        if prune_keep is not None:
            with torch.no_grad():
                generated = model.generate_params()
                if args.source_pruning_masks:
                    fixed_masks = load_pruning_masks(
                        Path(args.source_pruning_masks), device, generated
                    )
                else:
                    _, fixed_masks = masked_params(
                        generated,
                        prune_keep,
                        scope=args.pruning_scope,
                    )
        _, best_acc = evaluate_mapping(
            model,
            test_loader,
            device,
            args.max_test_batches,
            fixed_masks=fixed_masks,
        )
        initial_pruned_acc = best_acc
        print(f"loaded_dense_test        = {dense_acc:.2%}")
        print(f"loaded_addon_test        = {best_acc:.2%}")
        if prune_keep is not None and args.prune_finetune_epochs > 0:
            optimizer = torch.optim.AdamW(
                optimized,
                lr=args.prune_finetune_lr,
                weight_decay=args.weight_decay,
            )
            torch.save(model.state_dict(), run_dir / "best_mapping_model.pt")
            for epoch in range(1, args.prune_finetune_epochs + 1):
                model.train()
                total_loss = 0.0
                total_task = 0.0
                total_correct = 0
                total_count = 0
                started = time.time()
                progress = tqdm(
                    train_loader,
                    desc=f"prune-ft {epoch}/{args.prune_finetune_epochs}",
                    leave=False,
                    disable=args.no_progress,
                )
                for index, (x, y) in enumerate(progress):
                    if args.max_train_batches is not None and index >= args.max_train_batches:
                        break
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss, logits, task = mapping_batch_loss(
                        args,
                        model,
                        coefficients,
                        x,
                        y,
                        fixed_masks=fixed_masks,
                    )
                    loss.backward()
                    if args.grad_clip is not None:
                        nn.utils.clip_grad_norm_(optimized, args.grad_clip)
                    optimizer.step()
                    total_loss += loss.item() * y.numel()
                    total_task += task * y.numel()
                    total_correct += int((logits.argmax(1) == y).sum())
                    total_count += y.numel()
                test_loss, test_acc = evaluate_mapping(
                    model,
                    test_loader,
                    device,
                    args.max_test_batches,
                    fixed_masks=fixed_masks,
                )
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    torch.save(
                        model.state_dict(), run_dir / "best_mapping_model.pt"
                    )
                row = {
                    "finetune_epoch": epoch,
                    "train_loss": total_loss / total_count,
                    "train_task_loss": total_task / total_count,
                    "train_acc": total_correct / total_count,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best_test_acc": best_acc,
                    "best_epoch": best_epoch,
                    "effective_nonzero_target_params": prune_keep,
                    "mapping_loss_coefficients": (
                        None
                        if coefficients is None
                        else coefficients.detached_values()
                    ),
                    "seconds": time.time() - started,
                }
                append_jsonl(row, run_dir / "prune_history.jsonl")
                print(
                    f"prune_ft={epoch:03d} train_acc={row['train_acc']:.2%} "
                    f"test_acc={test_acc:.2%} best={best_acc:.2%}"
                )
            model.load_state_dict(
                load_state_dict(run_dir / "best_mapping_model.pt", device)
            )
    else:
        optimizer = torch.optim.AdamW(
            optimized, lr=args.lr, weight_decay=args.weight_decay
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_task = 0.0
            total_correct = 0
            total_count = 0
            started = time.time()
            progress = tqdm(
                train_loader,
                desc=f"epoch {epoch}/{args.epochs}",
                leave=False,
                disable=args.no_progress,
            )
            for index, (x, y) in enumerate(progress):
                if args.max_train_batches is not None and index >= args.max_train_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, logits, task = mapping_batch_loss(
                    args, model, coefficients, x, y
                )
                loss.backward()
                if args.grad_clip is not None:
                    nn.utils.clip_grad_norm_(optimized, args.grad_clip)
                optimizer.step()
                total_loss += loss.item() * y.numel()
                total_task += task * y.numel()
                total_correct += int((logits.argmax(1) == y).sum())
                total_count += y.numel()
            _, dense_acc = evaluate_mapping(
                model, test_loader, device, args.max_test_batches
            )
            test_loss, addon_acc = evaluate_mapping(
                model,
                test_loader,
                device,
                args.max_test_batches,
                prune_keep,
                pruning_scope=args.pruning_scope,
            )
            selection_acc = dense_acc if prune_keep is not None else addon_acc
            if selection_acc > best_acc:
                best_acc = selection_acc
                best_epoch = epoch
                checkpoint_name = (
                    "best_dense_mapping_model.pt"
                    if prune_keep is not None
                    else "best_mapping_model.pt"
                )
                torch.save(model.state_dict(), run_dir / checkpoint_name)
                if coefficients is not None:
                    torch.save(
                        coefficients.state_dict(),
                        run_dir / "best_mapping_loss_coefficients.pt",
                    )
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_count,
                "train_task_loss": total_task / total_count,
                "train_acc": total_correct / total_count,
                "dense_test_acc": dense_acc,
                "test_loss": test_loss,
                "test_acc": addon_acc,
                "best_test_acc": best_acc,
                "best_epoch": best_epoch,
                "mapping_loss_coefficients": (
                    None if coefficients is None else coefficients.detached_values()
                ),
                "seconds": time.time() - started,
            }
            append_jsonl(row, run_dir / "history.jsonl")
            print(
                f"epoch={epoch:03d} train_acc={row['train_acc']:.2%} "
                f"dense={dense_acc:.2%} addon={addon_acc:.2%} best={best_acc:.2%}"
            )
        checkpoint_name = (
            "best_dense_mapping_model.pt"
            if prune_keep is not None
            else "best_mapping_model.pt"
        )
        model.load_state_dict(load_state_dict(run_dir / checkpoint_name, device))
        _, dense_acc = evaluate_mapping(
            model, test_loader, device, args.max_test_batches
        )
        if prune_keep is not None:
            with torch.no_grad():
                _, fixed_masks = masked_params(
                    model.generate_params(),
                    prune_keep,
                    scope=args.pruning_scope,
                )
            _, initial_pruned_acc = evaluate_mapping(
                model,
                test_loader,
                device,
                args.max_test_batches,
                fixed_masks=fixed_masks,
            )
            print(f"best_dense_test          = {dense_acc:.2%}")
            print(f"initial_pruned_test      = {initial_pruned_acc:.2%}")
            best_acc, best_epoch = finetune_pruned_mapping(
                args,
                model,
                coefficients,
                optimized,
                train_loader,
                test_loader,
                device,
                fixed_masks,
                run_dir,
                initial_pruned_acc,
            )

    generated = model.generate_params()
    effective_nonzero = count_specs(model.specs)
    if prune_keep is not None:
        if fixed_masks is None:
            generated, fixed_masks = masked_params(
                generated, prune_keep, scope=args.pruning_scope
            )
        else:
            generated = apply_tensor_masks(generated, fixed_masks)
        effective_nonzero = count_nonzero_tensors(generated)
        torch.save(
            {name: mask.cpu() for name, mask in fixed_masks.items()},
            run_dir / "pruning_masks.pt",
        )

    return {
        "best_epoch": best_epoch,
        "dense_test_acc": dense_acc,
        "initial_pruned_test_acc": (
            initial_pruned_acc
        ),
        "best_test_acc": best_acc,
        "effective_nonzero_target_params": effective_nonzero,
        "trainable_model_params": latent_count,
        "stored_model_params": latent_count,
        "trainable_loss_coefficients": (
            0 if coefficients is None else count_trainable_params(coefficients)
        ),
        "final_mapping_loss_coefficients": (
            None if coefficients is None else coefficients.detached_values()
        ),
        "layerwise_dims": model.layer_dims,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.variant.endswith("lrd") and args.lrd_rank != TABLE_COUNT_MATCHING_RANK:
        actual = count_specs(low_rank_specs(args.lrd_rank))
        raise ValueError(
            f"Table 8 reports {TABLE_LRD_PARAMS:,} parameters, which implies rank "
            f"{TABLE_COUNT_MATCHING_RANK}; rank {args.lrd_rank} gives {actual:,}. "
            f"The prose says rank {PAPER_PROSE_RANK}, an internal paper inconsistency."
        )
    if args.source_checkpoint and not args.variant.endswith("prune"):
        raise ValueError("--source-checkpoint is currently only for post-training pruning")
    if args.source_pruning_masks and not args.source_checkpoint:
        raise ValueError("--source-pruning-masks requires --source-checkpoint")
    if args.source_pruning_masks and not args.variant.endswith("prune"):
        raise ValueError("--source-pruning-masks is only valid for pruning variants")
    if args.source_pruning_masks and args.variant.startswith("cnn2"):
        raise ValueError(
            "--source-pruning-masks resume is currently for mapping variants"
        )
    if args.variant.startswith("cnn2") and args.loss_mode == "full":
        raise ValueError("Direct CNN2 add-ons use task loss")
    if args.variant.startswith("lwt"):
        if args.layerwise_latent_dims is None:
            args.layerwise_latent_dims = [268, 622, 1347, 451]
        if sum(args.layerwise_latent_dims) != 2_688:
            raise ValueError("LWT layerwise latent dimensions must sum to 2,688")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    seed_everything(args.seed)
    table_params, paper_acc = PAPER_TABLE_8[(args.variant, args.dataset)]
    run_name = (
        f"{now_string()}_table8_{args.variant}_{args.dataset}"
        f"_p{table_params}_seed{args.seed}"
    )
    run_dir = ensure_dir(Path(args.results_dir) / run_name)
    device = resolve_device(args.device)
    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "device_resolved": str(device),
            "paper_acc": paper_acc,
            "table_params": table_params,
            "paper_rank_statement": PAPER_PROSE_RANK,
            "count_matching_rank": TABLE_COUNT_MATCHING_RANK,
            "rank_inconsistency_note": (
                "Paper prose says rank 16, but 35,914 parameters exactly imply "
                "rank 32 when only CNN2 FC1 is factorized."
            ),
            "pruning_definition": (
                f"{args.pruning_scope}-scope post-training magnitude pruning "
                "over weights and biases; "
                "keep exactly 10,862 of 108,618 generated/target parameters. "
                "Optional fine-tuning keeps the mask fixed and blocks gradients "
                "for pruned entries."
            ),
        }
    )
    save_json(config, run_dir / "config.json")

    print("=" * 80)
    print("Paper 3.6 Table 8 add-ons reproduction")
    print("=" * 80)
    print(f"run_dir                  = {run_dir}")
    print(f"device                   = {device}")
    print(f"variant                  = {args.variant}")
    print(f"dataset                  = {args.dataset}")
    print(f"table_params             = {table_params:,}")
    print(f"paper_acc                = {paper_acc:.2%}")
    if args.variant.endswith("lrd"):
        print(f"lrd_rank                 = {args.lrd_rank} (table-count matching)")
        print(f"paper_prose_rank         = {PAPER_PROSE_RANK} (inconsistent)")
    print("=" * 80)

    if args.variant.startswith("cnn2"):
        result = run_direct(args, run_dir, paper_acc)
    else:
        result = run_mapping(args, run_dir, paper_acc)

    final = {
        "run_name": run_name,
        "variant": args.variant,
        "dataset": args.dataset,
        "table_params": table_params,
        "best_epoch": result["best_epoch"],
        "best_test_acc": result["best_test_acc"],
        "paper_acc": paper_acc,
        "gap_best_minus_paper": result["best_test_acc"] - paper_acc,
        "dense_test_acc": result["dense_test_acc"],
        "initial_pruned_test_acc": result.get("initial_pruned_test_acc"),
        "effective_nonzero_target_params": result[
            "effective_nonzero_target_params"
        ],
        "trainable_model_params": result["trainable_model_params"],
        "stored_model_params": result["stored_model_params"],
        "trainable_loss_coefficients": result[
            "trainable_loss_coefficients"
        ],
        "total_optimized_params": (
            result["trainable_model_params"]
            + result["trainable_loss_coefficients"]
        ),
        "final_mapping_loss_coefficients": result[
            "final_mapping_loss_coefficients"
        ],
        "layerwise_dims": result.get("layerwise_dims"),
        "lrd_rank": args.lrd_rank if args.variant.endswith("lrd") else None,
        "paper_prose_rank": (
            PAPER_PROSE_RANK if args.variant.endswith("lrd") else None
        ),
        "implementation_note": config["rank_inconsistency_note"]
        if args.variant.endswith("lrd")
        else config["pruning_definition"],
    }
    save_json(final, run_dir / "final_result.json")
    print("=" * 80)
    print(f"best_test_acc            = {final['best_test_acc']:.2%}")
    print(f"paper_acc                = {paper_acc:.2%}")
    print(f"gap                      = {final['gap_best_minus_paper']:+.2%}")
    print(f"saved to                 = {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce paper Section 3.6 Table 8 add-on experiments"
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--dataset", choices=["mnist", "fmnist"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/table8")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--source-pruning-masks")
    parser.add_argument("--prune-finetune-epochs", type=int, default=0)
    parser.add_argument("--prune-finetune-lr", type=float, default=1e-4)
    parser.add_argument(
        "--pruning-scope",
        choices=["global", "tensor"],
        default="global",
    )

    parser.add_argument("--lrd-rank", type=int, default=TABLE_COUNT_MATCHING_RANK)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--activation", choices=["identity", "tanh", "sin"], default="identity")
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--modulation-scale", type=float, default=0.01)
    parser.add_argument("--latent-init-std", type=float, default=1.0)
    parser.add_argument("--projection-init", choices=["orthogonal", "gaussian"], default="orthogonal")
    parser.add_argument("--projection-layout", choices=["global", "blockwise"], default="global")
    parser.add_argument("--modulation-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--layerwise-latent-dims", type=int, nargs="+")
    parser.add_argument("--layerwise-modulation-scales", type=float, nargs="+")
    parser.add_argument("--stability-sigma", type=float, default=0.01)
    parser.add_argument("--loss-mode", choices=["task", "full"])
    parser.add_argument("--lambda-stability", type=float, default=0.05)
    parser.add_argument("--lambda-smoothness", type=float, default=5e-6)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.variant.startswith("cnn2") else 1e-2
    if args.loss_mode is None:
        args.loss_mode = "task" if args.variant.startswith("cnn2") else "full"
    run(args)


if __name__ == "__main__":
    main()
