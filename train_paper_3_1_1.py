from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ============================================================
# 0. 论文 3.1.1 Table 1 目标值
# ============================================================

PAPER_TABLE_1 = {
    ("baseline", "cnn1", "mnist", None): 0.9932,
    ("baseline", "cnn1", "fmnist", None): 0.9289,

    ("slvt", "cnn1", "mnist", 1024): 0.9878,
    ("slvt", "cnn1", "fmnist", 1024): 0.9302,
    ("slvt", "cnn1", "mnist", 2072): 0.9956,
    ("slvt", "cnn1", "fmnist", 2072): 0.9391,

    ("lwt", "cnn1", "mnist", 4078): 0.9967,
    ("lwt", "cnn1", "fmnist", 4078): 0.9483,

    ("baseline", "cnn2", "mnist", None): 0.9869,
    ("baseline", "cnn2", "fmnist", None): 0.9040,

    ("slvt", "cnn2", "mnist", 1024): 0.9788,
    ("slvt", "cnn2", "fmnist", 1024): 0.8949,
    ("slvt", "cnn2", "mnist", 2048): 0.9866,
    ("slvt", "cnn2", "fmnist", 2048): 0.9188,

    ("lwt", "cnn2", "mnist", 1872): 0.9898,
    ("lwt", "cnn2", "fmnist", 1872): 0.9284,
    ("lwt", "cnn2", "mnist", 2688): 0.9918,
    ("lwt", "cnn2", "fmnist", 2688): 0.9335,
}

MAPPING_LOSS_TERMS = ("stability", "smoothness", "alignment")

PAPER_UNDISCLOSED_SETTINGS = (
    "activation",
    "modulation_scale",
    "layerwise_modulation_scales",
    "latent_init_std",
    "layerwise_latent_dims",
    "epochs",
    "batch_size",
    "lr",
    "optimizer",
    "data_normalization",
    "mapping_loss_coefficient_initialization",
)

PAPER_TABLE_6_CNN2_FMNIST_1872 = {
    (): 0.8911,
    ("stability",): 0.8956,
    ("smoothness",): 0.8943,
    ("alignment",): 0.8932,
    ("alignment", "smoothness"): 0.9047,
    ("smoothness", "stability"): 0.9111,
    ("alignment", "smoothness", "stability"): 0.9284,
}


# ============================================================
# 1. 基础工具函数
# ============================================================

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    固定随机种子，尽量保证实验可复现。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def numel_from_shape(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def now_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: Path) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(obj: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# 2. 数据加载
# ============================================================

def get_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    加载 MNIST / FashionMNIST。

    论文 3.1.1 用的是：
    - MNIST
    - FashionMNIST

    这里使用 torchvision 官方 train/test split。
    """

    dataset_name = dataset_name.lower()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
    elif dataset_name == "fmnist":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_set = dataset_cls(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = dataset_cls(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


# ============================================================
# 3. CNN1 / CNN2 baseline
# ============================================================

class CNN1(nn.Module):
    """
    CNN1: AlexNet-inspired compact CNN.

    输入:
        1 x 28 x 28

    参数量:
        537,994

    结构:
        Conv1: 1 -> 32, 3x3, padding=1
        MaxPool
        Conv2: 32 -> 64, 3x3, padding=1
        MaxPool
        Conv3: 64 -> 128, 3x3, padding=1
        Conv4: 128 -> 128, 3x3, padding=1
        MaxPool
        FC1: 128*3*3 -> 256
        FC2: 256 -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNN2(nn.Module):
    """
    CNN2: LeNet-inspired compact CNN.

    输入:
        1 x 28 x 28

    参数量:
        108,618

    结构:
        Conv1: 1 -> 16, 3x3, padding=0
        MaxPool
        Conv2: 16 -> 32, 3x3, padding=0
        MaxPool
        FC1: 32*5*5 -> 128
        FC2: 128 -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def make_baseline(model_name: str) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "cnn1":
        return CNN1()

    if model_name == "cnn2":
        return CNN2()

    raise ValueError(f"Unknown model: {model_name}")


def get_param_specs(model_name: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    获取目标 CNN 的参数名称与 shape。

    Mapping CNN 会生成这些参数，而不是直接训练它们。
    """

    model = make_baseline(model_name)

    specs = []

    for name, param in model.named_parameters():
        specs.append((name, tuple(param.shape)))

    return specs


def count_params_from_specs(specs: Iterable[Tuple[str, Tuple[int, ...]]]) -> int:
    total = 0

    for _, shape in specs:
        total += numel_from_shape(shape)

    return total


# ============================================================
# 4. 使用生成参数进行 functional forward
# ============================================================

def functional_cnn_forward(
    x: Tensor,
    model_name: str,
    params: Dict[str, Tensor],
) -> Tensor:
    """
    用 Mapping Network 生成的 params 做 CNN 前向传播。

    注意：
    这里没有 nn.Conv2d 的可训练参数。
    conv/fc 参数全部来自 params。
    """

    model_name = model_name.lower()

    if model_name == "cnn1":
        x = F.conv2d(
            x,
            params["conv1.weight"],
            params["conv1.bias"],
            padding=1,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(
            x,
            params["conv2.weight"],
            params["conv2.bias"],
            padding=1,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(
            x,
            params["conv3.weight"],
            params["conv3.bias"],
            padding=1,
        )
        x = F.relu(x)

        x = F.conv2d(
            x,
            params["conv4.weight"],
            params["conv4.bias"],
            padding=1,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = F.linear(
            x,
            params["fc1.weight"],
            params["fc1.bias"],
        )
        x = F.relu(x)

        x = F.linear(
            x,
            params["fc2.weight"],
            params["fc2.bias"],
        )

        return x

    if model_name == "cnn2":
        x = F.conv2d(
            x,
            params["conv1.weight"],
            params["conv1.bias"],
            padding=0,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(
            x,
            params["conv2.weight"],
            params["conv2.bias"],
            padding=0,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = F.linear(
            x,
            params["fc1.weight"],
            params["fc1.bias"],
        )
        x = F.relu(x)

        x = F.linear(
            x,
            params["fc2.weight"],
            params["fc2.bias"],
        )

        return x

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 5. Mapping CNN
# ============================================================

def allocate_layerwise_dims(
    specs: List[Tuple[str, Tuple[int, ...]]],
    total_latent_dim: int,
    min_dim: int = 4,
) -> Dict[str, int]:
    """
    Ours† / LWT 每一层 latent dimension 的分配规则。

    论文没有公开每一层 latent vector 的具体分配。
    这里采用一个透明、可复现的规则：

    1. 按每个参数张量大小占比分配；
    2. 每个参数张量至少 min_dim；
    3. 最后调整到总 latent 维度正好等于 total_latent_dim。
    """

    if total_latent_dim < len(specs) * min_dim:
        raise ValueError(
            f"total_latent_dim={total_latent_dim} 太小，"
            f"至少需要 {len(specs) * min_dim}"
        )

    sizes = {name: numel_from_shape(shape) for name, shape in specs}
    total_size = sum(sizes.values())

    dims = {}
    remaining = total_latent_dim

    for name, _ in specs:
        dim = round(total_latent_dim * sizes[name] / total_size)
        dim = max(min_dim, int(dim))

        dims[name] = dim
        remaining -= dim

    order = sorted(
        specs,
        key=lambda item: sizes[item[0]],
        reverse=True,
    )

    idx = 0

    while remaining != 0:
        name = order[idx % len(order)][0]

        if remaining > 0:
            dims[name] += 1
            remaining -= 1
        else:
            if dims[name] > min_dim:
                dims[name] -= 1
                remaining += 1

        idx += 1

    assert sum(dims.values()) == total_latent_dim

    return dims


def validate_layerwise_dims(
    specs: List[Tuple[str, Tuple[int, ...]]],
    total_latent_dim: int,
    requested_dims: List[int],
) -> Dict[str, int]:
    if len(requested_dims) != len(specs):
        raise ValueError(
            f"Expected {len(specs)} layer-wise latent dimensions, "
            f"got {len(requested_dims)}"
        )
    if any(dim <= 0 for dim in requested_dims):
        raise ValueError("Layer-wise latent dimensions must all be positive")
    if sum(requested_dims) != total_latent_dim:
        raise ValueError(
            f"Layer-wise latent dimensions sum to {sum(requested_dims)}, "
            f"expected {total_latent_dim}"
        )
    return {
        name: dim
        for (name, _), dim in zip(specs, requested_dims)
    }


def validate_layerwise_modulation_scales(
    layer_names: Iterable[str],
    default_scale: float,
    requested_scales: Optional[List[float]],
) -> Dict[str, float]:
    names = list(layer_names)
    if requested_scales is None:
        scales = [default_scale] * len(names)
    else:
        if len(requested_scales) != len(names):
            raise ValueError(
                f"Expected {len(names)} layer-wise modulation scales, "
                f"got {len(requested_scales)}"
            )
        scales = requested_scales

    if any(scale <= 0 for scale in scales):
        raise ValueError("Layer-wise modulation scales must all be positive")

    return {
        name: float(scale)
        for name, scale in zip(names, scales)
    }


class ChunkedFixedProjector(nn.Module):
    """
    固定随机投影生成器。

    输入:
        latent vector z, shape = [latent_dim]

    输出:
        一个扁平化的目标网络参数向量 theta, shape = [out_dim]

    关键设计：
        1. z 是唯一可训练参数；
        2. 投影矩阵 W 不训练；
        3. 为避免存储巨大 W，按 chunk 动态生成固定随机 W；
        4. 通过 seed + chunk 位置保证每次生成的 W 一致。

    这对应论文 / 仓库共同的核心机制：
        small trainable latent -> fixed projection/mapping -> generated CNN weights
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        seed: int,
        chunk_size: int = 4096,
        activation: str = "tanh",
        weight_scale: float = 0.07,
        modulation_scale: float = 0.01,
        latent_init_std: float = 0.02,
        projection_init: str = "orthogonal",
        modulation_reduction: str = "sum",
        projection_layout: str = "global",
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.activation = activation
        self.weight_scale = float(weight_scale)
        self.modulation_scale = float(modulation_scale)
        self.projection_init = projection_init
        self.modulation_reduction = modulation_reduction
        self.projection_layout = projection_layout

        if self.projection_layout not in {"global", "blockwise"}:
            raise ValueError(
                f"Unknown projection layout: {self.projection_layout}"
            )

        self.z = nn.Parameter(
            torch.randn(self.latent_dim) * latent_init_std
        )
        # 对 CNN2 这种中小模型，直接缓存完整固定投影矩阵，可以大幅加速。
        # W 和 b 是 buffer，不是 Parameter，不会被训练。
        W_cpu, b_cpu = self._make_projection_cpu()

        self.register_buffer(
            "cached_W",
            W_cpu,
            persistent=False,
        )

        self.register_buffer(
            "cached_b",
            b_cpu,
            persistent=False,
        )

        # These projection statistics are constant. Precomputing them avoids
        # scanning the large fixed matrix for every Mapping Loss batch.
        if self.activation == "identity" and self.projection_init == "orthogonal":
            column_norm_sq = torch.empty(0, dtype=torch.float32)
            if self.projection_layout == "global":
                orthogonal_rank = min(self.latent_dim, self.out_dim)
            else:
                orthogonal_rank = sum(
                    min(self.latent_dim, end - start)
                    for start in range(0, self.out_dim, self.chunk_size)
                    for end in [min(start + self.chunk_size, self.out_dim)]
                )
            frobenius_norm_sq = torch.tensor(
                float(orthogonal_rank),
                dtype=torch.float32,
            )
        else:
            column_norm_sq = torch.empty(self.out_dim, dtype=torch.float32)
            for start in range(0, self.out_dim, self.chunk_size):
                end = min(start + self.chunk_size, self.out_dim)
                block = W_cpu[:, start:end]
                column_norm_sq[start:end] = torch.sum(block * block, dim=0)
            frobenius_norm_sq = torch.sum(column_norm_sq)

        self.register_buffer(
            "cached_W_column_norm_sq",
            column_norm_sq,
            persistent=False,
        )
        self.register_buffer(
            "cached_W_frobenius_norm_sq",
            frobenius_norm_sq,
            persistent=False,
        )
        self.register_buffer(
            "cached_W_row_mean",
            W_cpu.mean(dim=1),
            persistent=False,
        )

    def _make_projection_cpu(self) -> Tuple[Tensor, Tensor]:
        if self.projection_layout == "global":
            return self._make_chunk_cpu(0, self.out_dim)

        W = torch.empty(
            self.latent_dim,
            self.out_dim,
            dtype=torch.float32,
        )
        b = torch.empty(self.out_dim, dtype=torch.float32)

        for start in range(0, self.out_dim, self.chunk_size):
            end = min(start + self.chunk_size, self.out_dim)
            W_chunk, b_chunk = self._make_chunk_cpu(start, end)
            W[:, start:end].copy_(W_chunk)
            b[start:end].copy_(b_chunk)

        return W, b

    def _make_chunk_cpu(
        self,
        start: int,
        end: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        生成固定随机投影矩阵的一个 chunk。

        注意：
        每次用相同 seed + start + end 生成，所以 W 是固定的。
        但是 W 不作为可训练参数保存。
        """

        width = end - start

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed
            + start * 1009
            + end * 9176
            + self.latent_dim * 37
        )

        W = torch.empty(
            self.latent_dim,
            width,
            dtype=torch.float32,
        )

        if self.projection_init == "orthogonal":
            nn.init.orthogonal_(W, generator=generator)
        elif self.projection_init == "gaussian":
            W.normal_(generator=generator)
            W = W / math.sqrt(self.latent_dim)
        else:
            raise ValueError(
                f"Unknown projection initialization: {self.projection_init}"
            )

        b = torch.randn(
            width,
            generator=generator,
            dtype=torch.float32,
        ) * 0.01

        return W, b

    def _activate(self, x: Tensor) -> Tensor:
        if self.activation == "tanh":
            return torch.tanh(x)

        if self.activation == "identity":
            return x

        if self.activation == "sin":
            return torch.sin(x)

        raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, raw: Tensor) -> Tensor:
        if self.activation == "tanh":
            activated = torch.tanh(raw)
            return 1.0 - activated * activated
        if self.activation == "identity":
            return torch.ones_like(raw)
        if self.activation == "sin":
            return torch.cos(raw)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _modulation_coefficient(self) -> float:
        if self.modulation_reduction == "sum":
            return self.modulation_scale
        if self.modulation_reduction == "mean":
            return self.modulation_scale / self.latent_dim
        raise ValueError(
            f"Unknown modulation reduction: {self.modulation_reduction}"
        )

    def _raw_mapping_output(self, z: Tensor) -> Tensor:
        W = self.cached_W.to(device=z.device, dtype=z.dtype)
        b = self.cached_b.to(device=z.device, dtype=z.dtype)
        beta = self._modulation_coefficient()
        return torch.matmul(z, W) + beta * torch.sum(z * z) + b

    def forward(self, z_override: Optional[Tensor] = None) -> Tensor:
        z = self.z if z_override is None else z_override

        theta = self._activate(self._raw_mapping_output(z))
        theta = theta * self.weight_scale
        
        return theta

    def smoothness_loss(self) -> Tensor:
        """Exact squared Frobenius norm of the mapping Jacobian."""
        z = self.z
        beta = self._modulation_coefficient()
        sum_z_sq = torch.sum(z * z)
        column_norm_sq = self.cached_W_column_norm_sq.to(
            device=z.device,
            dtype=z.dtype,
        )
        row_mean = self.cached_W_row_mean.to(device=z.device, dtype=z.dtype)

        if self.activation == "identity":
            projected_sum = self.out_dim * torch.dot(z, row_mean)
            jacobian_norm_sq = (
                self.cached_W_frobenius_norm_sq.to(
                    device=z.device,
                    dtype=z.dtype,
                )
                + 4.0 * beta * projected_sum
                + 4.0 * beta * beta * self.out_dim * sum_z_sq
            )
            return self.weight_scale * self.weight_scale * torch.clamp_min(
                jacobian_norm_sq,
                0.0,
            )

        raw = self._raw_mapping_output(z)
        activation_grad = self._activation_derivative(raw) * self.weight_scale
        b = self.cached_b.to(device=z.device, dtype=z.dtype)
        projected_z = raw - beta * sum_z_sq - b

        # For raw_j = sum_i(W_ij z_i) + beta * sum_i(z_i^2) + b_j,
        # sum_i (d raw_j / d z_i)^2 can be evaluated without materializing J.
        column_jacobian_norm_sq = (
            column_norm_sq
            + 4.0 * beta * projected_z
            + 4.0 * beta * beta * sum_z_sq
        )
        column_jacobian_norm_sq = torch.clamp_min(column_jacobian_norm_sq, 0.0)
        return torch.sum(activation_grad * activation_grad * column_jacobian_norm_sq)

    def alignment_loss(self) -> Tensor:
        """1 - cosine(z, row_mean(W_modulated)) from the paper."""
        z = self.z
        beta = self._modulation_coefficient()
        row_mean = self.cached_W_row_mean.to(
            device=z.device,
            dtype=z.dtype,
        ) + beta * z

        cosine = F.cosine_similarity(
            z.unsqueeze(0),
            row_mean.unsqueeze(0),
            dim=1,
        ).mean()

        return 1.0 - cosine



class MappingCNN(nn.Module):
    """
    Mapping CNN for paper 3.1.1.

    mode:
        slvt:
            Ours*
            一个 latent vector 生成整个 CNN 的所有参数。

        lwt:
            Ours†
            每个参数张量有自己的 latent vector。
    """
 
    def _build_layer_groups(self) -> Dict[str, List[Tuple[str, Tuple[int, ...]]]]:
        """
        真正的 layer-wise 分组。

        CNN2:
            conv1.weight + conv1.bias
            conv2.weight + conv2.bias
            fc1.weight + fc1.bias
            fc2.weight + fc2.bias

        CNN1:
            conv1.weight + conv1.bias
            conv2.weight + conv2.bias
            conv3.weight + conv3.bias
            conv4.weight + conv4.bias
            fc1.weight + fc1.bias
            fc2.weight + fc2.bias
        """

        groups: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = OrderedDict()

        for name, shape in self.specs:
            layer_name = name.split(".")[0]

            if layer_name not in groups:
                groups[layer_name] = []

            groups[layer_name].append((name, shape))

        return groups
    def __init__(
        self,
        model_name: str,
        mode: str,
        latent_dim: Optional[int],
        total_latent_dim: Optional[int],
        seed: int,
        chunk_size: int,
        activation: str,
        weight_scale: float,
        modulation_scale: float,
        latent_init_std: float,
        projection_init: str,
        modulation_reduction: str,
        parameter_scale_mode: str,
        layerwise_latent_dims: Optional[List[int]],
        layerwise_modulation_scales: Optional[List[float]],
        projection_layout: str = "global",
    ):
        super().__init__()

        self.model_name = model_name.lower()
        self.mode = mode.lower()
        self.parameter_scale_mode = parameter_scale_mode

        self.specs = get_param_specs(self.model_name)
        self.target_param_count = count_params_from_specs(self.specs)

        if self.mode == "slvt":
            if latent_dim is None:
                raise ValueError("SLVT / Ours* requires --latent-dim")
            if layerwise_modulation_scales is not None:
                raise ValueError(
                    "--layerwise-modulation-scales is only valid for LWT / Ours†"
                )

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

        elif self.mode == "lwt":
            if total_latent_dim is None:
                raise ValueError("LWT / Ours† requires --total-latent-dim")

            # 真正按 layer 分组，而不是按每个 weight/bias 张量分组。
            self.layer_groups = self._build_layer_groups()

            group_specs: List[Tuple[str, Tuple[int, ...]]] = []

            for group_name, items in self.layer_groups.items():
                group_out_dim = sum(
                    numel_from_shape(shape)
                    for _, shape in items
                )
                group_specs.append((group_name, (group_out_dim,)))

            if layerwise_latent_dims is None:
                self.layer_dims = allocate_layerwise_dims(
                    specs=group_specs,
                    total_latent_dim=total_latent_dim,
                    min_dim=32,
                )
            else:
                self.layer_dims = validate_layerwise_dims(
                    specs=group_specs,
                    total_latent_dim=total_latent_dim,
                    requested_dims=layerwise_latent_dims,
                )

            self.layer_modulation_scales = validate_layerwise_modulation_scales(
                layer_names=self.layer_groups.keys(),
                default_scale=modulation_scale,
                requested_scales=layerwise_modulation_scales,
            )

            self.projectors = nn.ModuleDict()

            for i, (group_name, items) in enumerate(self.layer_groups.items()):
                group_out_dim = sum(
                    numel_from_shape(shape)
                    for _, shape in items
                )

                self.projectors[group_name] = ChunkedFixedProjector(
                    latent_dim=self.layer_dims[group_name],
                    out_dim=group_out_dim,
                    seed=seed + 100000 + i * 997,
                    chunk_size=chunk_size,
                    activation=activation,
                    weight_scale=weight_scale,
                    modulation_scale=self.layer_modulation_scales[group_name],
                    latent_init_std=latent_init_std,
                    projection_init=projection_init,
                    modulation_reduction=modulation_reduction,
                    projection_layout=projection_layout,
                )

        else:
            raise ValueError("mode must be slvt or lwt")

    def latent_parameters(self) -> List[nn.Parameter]:
        if self.mode == "slvt":
            return [self.projector.z]
        return [projector.z for projector in self.projectors.values()]

    def trainable_latent_count(self) -> int:
        return sum(parameter.numel() for parameter in self.latent_parameters())

    def _scale_generated_param(
        self,
        name: str,
        value: Tensor,
    ) -> Tensor:
        """
        对生成的参数做合理缩放，避免训练初期数值过大。

        这不是额外可训练参数，只是让 generated weights 的尺度接近普通初始化。
        """

        if self.parameter_scale_mode == "paper":
            return value

        if name.endswith("weight") and value.ndim >= 2:
            fan_in = numel_from_shape(tuple(value.shape[1:]))
            return value / math.sqrt(fan_in)

        if name.endswith("bias"):
            return value * 0.1

        return value

    def generate_params(
        self,
        z_perturb_scale: float = 0.0,
    ) -> Dict[str, Tensor]:
        params = OrderedDict()

        if self.mode == "slvt":
            if z_perturb_scale == 0.0:
                flat = self.projector()
            else:
                eps = torch.randn_like(self.projector.z) * z_perturb_scale
                flat = self.projector(self.projector.z + eps)

            offset = 0

            for name, shape in self.specs:
                n = numel_from_shape(shape)

                value = flat[offset: offset + n].view(shape)
                value = self._scale_generated_param(name, value)

                params[name] = value

                offset += n

            return params

        if self.mode == "lwt":
            for group_name, items in self.layer_groups.items():
                projector = self.projectors[group_name]

                if z_perturb_scale == 0.0:
                    group_flat = projector()
                else:
                    eps = torch.randn_like(projector.z) * z_perturb_scale
                    group_flat = projector(projector.z + eps)

                offset = 0

                for name, shape in items:
                    n = numel_from_shape(shape)

                    value = group_flat[offset: offset + n].view(shape)
                    value = self._scale_generated_param(name, value)

                    params[name] = value

                    offset += n

            return params

        raise RuntimeError("Unknown mode")

    def smoothness_loss(self) -> Tensor:
        if self.mode == "slvt":
            return self.projector.smoothness_loss()
        return torch.stack(
            [projector.smoothness_loss() for projector in self.projectors.values()]
        ).sum()

    def alignment_loss(self) -> Tensor:
        if self.mode == "slvt":
            return self.projector.alignment_loss()

        losses = []

        for projector in self.projectors.values():
            losses.append(projector.alignment_loss())

        return torch.stack(losses).mean()

    def forward(self, x: Tensor) -> Tensor:
        params = self.generate_params(z_perturb_scale=0.0)

        return functional_cnn_forward(
            x=x,
            model_name=self.model_name,
            params=params,
        )


# ============================================================
# 6. Baseline 训练与评估
# ============================================================

@torch.no_grad()
def evaluate_baseline(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.numel()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += y.numel()

    return total_loss / total_count, total_correct / total_count


def train_baseline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    device = resolve_device(args.device)

    train_loader, test_loader = get_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = make_baseline(args.model).to(device)

    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    paper_acc = PAPER_TABLE_1.get(
        ("baseline", args.model, args.dataset, None)
    )

    run_name = (
        f"{now_string()}_baseline_{args.model}_{args.dataset}"
        f"_seed{args.seed}"
    )

    run_dir = ensure_dir(Path(args.results_dir) / run_name)

    config = vars(args).copy()
    config.update(
        {
            "experiment_type": "baseline",
            "run_name": run_name,
            "device_resolved": str(device),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "paper_table_1_acc": paper_acc,
        }
    )

    save_json(config, run_dir / "config.json")

    print("=" * 80)
    print("Paper 3.1.1 baseline reproduction")
    print("=" * 80)
    print(f"run_dir             = {run_dir}")
    print(f"device              = {device}")
    print(f"dataset             = {args.dataset}")
    print(f"model               = {args.model}")
    print(f"trainable_params    = {trainable_params:,}")
    print(f"total_params        = {total_params:,}")
    print(f"paper_acc           = {paper_acc}")
    print("=" * 80)

    best_test_acc = 0.0
    best_epoch = 0

    history_path = run_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        )

        for batch_idx, (x, y) in enumerate(progress):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += y.numel()

            progress.set_postfix(
                train_loss=total_loss / total_count,
                train_acc=total_correct / total_count,
            )

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        test_loss, test_acc = evaluate_baseline(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.max_test_batches,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "paper_acc": paper_acc,
            "gap_best_minus_paper": None if paper_acc is None else best_test_acc - paper_acc,
        }

        append_jsonl(row, history_path)

        paper_display = "n/a" if paper_acc is None else f"{paper_acc * 100:.2f}%"
        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_acc * 100:.2f}% "
            f"test_acc={test_acc * 100:.2f}% "
            f"best={best_test_acc * 100:.2f}% "
            f"paper={paper_display}"
        )

    final_result = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "paper_acc": paper_acc,
        "gap_best_minus_paper": None if paper_acc is None else best_test_acc - paper_acc,
    }

    save_json(final_result, run_dir / "final_result.json")

    print("=" * 80)
    print("Finished baseline run")
    print(f"best_epoch          = {best_epoch}")
    print(f"best_test_acc       = {best_test_acc * 100:.2f}%")
    if paper_acc is not None:
        print(f"paper_acc           = {paper_acc * 100:.2f}%")
        print(f"gap                 = {(best_test_acc - paper_acc) * 100:.2f}%")
    print(f"saved to            = {run_dir}")
    print("=" * 80)


# ============================================================
# 7. Mapping CNN 训练与评估
# ============================================================

@torch.no_grad()
def evaluate_mapping(
    model: MappingCNN,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    params = model.generate_params(z_perturb_scale=0.0)

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = functional_cnn_forward(
            x=x,
            model_name=model.model_name,
            params=params,
        )

        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.numel()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += y.numel()

    return total_loss / total_count, total_correct / total_count


def validate_mapping_loss_terms(terms: Iterable[str]) -> Tuple[str, ...]:
    selected = tuple(dict.fromkeys(terms))
    invalid = sorted(set(selected) - set(MAPPING_LOSS_TERMS))
    if invalid:
        raise ValueError(f"Unknown Mapping Loss terms: {invalid}")
    if not selected:
        raise ValueError("Full Mapping Loss requires at least one auxiliary term")
    return selected


def validate_paper_protocol(args: argparse.Namespace) -> None:
    """Enforce every training choice that the paper states explicitly."""
    if args.protocol != "paper":
        return

    if args.layerwise_modulation_scales is not None:
        raise ValueError(
            "The paper permits layer-specific modulation rates for fine tuning "
            "but does not report them for Table 1. Use --protocol custom for "
            "this controlled experiment."
        )

    required_values = {
        "projection_init": "orthogonal",
        "projection_layout": "global",
        "modulation_reduction": "sum",
        "parameter_scale_mode": "paper",
    }
    for name, expected in required_values.items():
        actual = getattr(args, name)
        if actual != expected:
            cli_name = name.replace("_", "-")
            raise ValueError(
                f"Paper protocol requires --{cli_name} {expected}; got {actual}"
            )

    if not math.isclose(args.weight_scale, 1.0):
        raise ValueError(
            "Paper protocol maps activated parameters directly and requires "
            "--weight-scale 1"
        )

    if args.loss_mode == "full":
        if args.loss_coefficient_mode != "trainable":
            raise ValueError(
                "Paper Equation (26) describes trainable Mapping Loss "
                "coefficients; use --loss-coefficient-mode trainable"
            )
        selected_terms = set(validate_mapping_loss_terms(args.mapping_loss_terms))
        if selected_terms != set(MAPPING_LOSS_TERMS):
            raise ValueError(
                "Paper Table 1 Full Mapping Loss requires stability, "
                "smoothness, and alignment. Use --protocol custom for "
                "Table 6 component ablations."
            )


class TrainableMappingLossCoefficients(nn.Module):
    """Positive trainable coefficients for the three Mapping Loss terms."""

    def __init__(
        self,
        stability: float,
        smoothness: float,
        alignment: float,
        enabled_terms: Iterable[str] = MAPPING_LOSS_TERMS,
    ):
        super().__init__()
        self.enabled_terms = validate_mapping_loss_terms(enabled_terms)
        initial_values = {
            "stability": stability,
            "smoothness": smoothness,
            "alignment": alignment,
        }
        self.raw_values = nn.ParameterDict(
            {
                name: nn.Parameter(self._inverse_softplus(initial_values[name]))
                for name in self.enabled_terms
            }
        )

    @staticmethod
    def _inverse_softplus(value: float) -> Tensor:
        if value <= 0:
            raise ValueError("Mapping Loss coefficient initial values must be positive")
        return torch.tensor(math.log(math.expm1(value)), dtype=torch.float32)

    def forward(self) -> Dict[str, Tensor]:
        reference = next(iter(self.raw_values.values()))
        return {
            name: (
                F.softplus(self.raw_values[name])
                if name in self.raw_values
                else reference.new_zeros(())
            )
            for name in MAPPING_LOSS_TERMS
        }

    def detached_values(self) -> Dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in self().items()
        }


class FixedMappingLossCoefficients(nn.Module):
    """Non-trainable coefficients used for controlled Mapping Loss ablations."""

    def __init__(
        self,
        stability: float,
        smoothness: float,
        alignment: float,
        enabled_terms: Iterable[str] = MAPPING_LOSS_TERMS,
    ):
        super().__init__()
        self.enabled_terms = validate_mapping_loss_terms(enabled_terms)
        initial_values = {
            "stability": stability,
            "smoothness": smoothness,
            "alignment": alignment,
        }
        for name in MAPPING_LOSS_TERMS:
            value = initial_values[name] if name in self.enabled_terms else 0.0
            if value < 0:
                raise ValueError("Fixed Mapping Loss coefficients must be non-negative")
            self.register_buffer(f"fixed_{name}", torch.tensor(value, dtype=torch.float32))

    def forward(self) -> Dict[str, Tensor]:
        return {
            name: getattr(self, f"fixed_{name}")
            for name in MAPPING_LOSS_TERMS
        }

    def detached_values(self) -> Dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in self().items()
        }


def compute_mapping_loss(
    args: argparse.Namespace,
    model: MappingCNN,
    loss_coefficients: Optional[nn.Module],
    x: Tensor,
    y: Tensor,
) -> Tuple[Tensor, Dict[str, float], Tensor]:
    """
    Mapping CNN loss。

    loss_mode:
        task:
            只使用分类交叉熵。

        full:
            task loss
            + stability loss
            + exact mapping-Jacobian smoothness loss
            + modulated-weight alignment loss
    """

    params = model.generate_params(z_perturb_scale=0.0)

    logits = functional_cnn_forward(
        x=x,
        model_name=model.model_name,
        params=params,
    )

    task_loss = F.cross_entropy(logits, y)

    if args.loss_mode == "task":
        parts = {
            "task_loss": float(task_loss.detach().cpu()),
            "stability_loss": 0.0,
            "smoothness_loss": 0.0,
            "alignment_loss": 0.0,
            "lambda_stability": 0.0,
            "lambda_smoothness": 0.0,
            "lambda_alignment": 0.0,
        }

        return task_loss, parts, logits

    if loss_coefficients is None:
        raise RuntimeError("Full Mapping Loss requires coefficients")

    enabled_terms = set(args.mapping_loss_terms)
    coefficients = loss_coefficients()
    zero = task_loss.new_zeros(())

    if "stability" in enabled_terms:
        params_perturbed = model.generate_params(
            z_perturb_scale=args.stability_sigma
        )
        logits_perturbed = functional_cnn_forward(
            x=x,
            model_name=model.model_name,
            params=params_perturbed,
        )
        stability_loss = torch.mean(
            torch.sum((logits_perturbed - logits) ** 2, dim=1)
        )
    else:
        stability_loss = zero

    smoothness_loss = (
        model.smoothness_loss()
        if "smoothness" in enabled_terms
        else zero
    )
    alignment_loss = (
        model.alignment_loss()
        if "alignment" in enabled_terms
        else zero
    )

    loss = (
        task_loss
        + coefficients["stability"] * stability_loss
        + coefficients["smoothness"] * smoothness_loss
        + coefficients["alignment"] * alignment_loss
    )

    parts = {
        "task_loss": float(task_loss.detach().cpu()),
        "stability_loss": float(stability_loss.detach().cpu()),
        "smoothness_loss": float(smoothness_loss.detach().cpu()),
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "lambda_stability": float(coefficients["stability"].detach().cpu()),
        "lambda_smoothness": float(coefficients["smoothness"].detach().cpu()),
        "lambda_alignment": float(coefficients["alignment"].detach().cpu()),
    }

    return loss, parts, logits


def train_mapping(args: argparse.Namespace) -> None:
    validate_paper_protocol(args)
    seed_everything(args.seed)

    if args.loss_mode == "full":
        if args.parameter_scale_mode != "paper":
            raise ValueError("Full paper Mapping Loss requires --parameter-scale-mode paper")
        args.mapping_loss_terms = list(
            validate_mapping_loss_terms(args.mapping_loss_terms)
        )

    device = resolve_device(args.device)

    train_loader, test_loader = get_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = MappingCNN(
        model_name=args.model,
        mode=args.mapping_mode,
        latent_dim=args.latent_dim,
        total_latent_dim=args.total_latent_dim,
        seed=args.seed,
        chunk_size=args.chunk_size,
        activation=args.activation,
        weight_scale=args.weight_scale,
        modulation_scale=args.modulation_scale,
        latent_init_std=args.latent_init_std,
        projection_init=args.projection_init,
        modulation_reduction=args.modulation_reduction,
        parameter_scale_mode=args.parameter_scale_mode,
        layerwise_latent_dims=args.layerwise_latent_dims,
        layerwise_modulation_scales=args.layerwise_modulation_scales,
        projection_layout=args.projection_layout,
    ).to(device)

    trainable_latent_params = model.trainable_latent_count()
    target_param_count = model.target_param_count

    loss_coefficients = None
    optimized_parameters = list(model.parameters())
    if args.loss_mode == "full":
        coefficient_class = (
            FixedMappingLossCoefficients
            if args.loss_coefficient_mode == "fixed"
            else TrainableMappingLossCoefficients
        )
        loss_coefficients = coefficient_class(
            stability=args.lambda_stability,
            smoothness=args.lambda_smoothness,
            alignment=args.lambda_alignment,
            enabled_terms=args.mapping_loss_terms,
        ).to(device)
        optimized_parameters.extend(loss_coefficients.parameters())

    trainable_loss_coefficients = (
        0
        if loss_coefficients is None
        else sum(p.numel() for p in loss_coefficients.parameters() if p.requires_grad)
    )

    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.mapping_mode == "slvt":
        z_key = args.latent_dim
    else:
        z_key = args.total_latent_dim

    if (
        args.mapping_mode == "lwt"
        and args.model == "cnn2"
        and args.dataset == "fmnist"
        and z_key == 1872
    ):
        table_6_terms = (
            ()
            if args.loss_mode == "task"
            else tuple(sorted(args.mapping_loss_terms))
        )
        paper_acc = PAPER_TABLE_6_CNN2_FMNIST_1872.get(table_6_terms)
    else:
        paper_acc = PAPER_TABLE_1.get(
            (args.mapping_mode, args.model, args.dataset, z_key)
        )

    run_name = (
        f"{now_string()}_mapping_{args.mapping_mode}_{args.model}_{args.dataset}"
        f"_z{z_key}_seed{args.seed}"
    )

    run_dir = ensure_dir(Path(args.results_dir) / run_name)

    config = vars(args).copy()
    config.update(
        {
            "experiment_type": "mapping",
            "run_name": run_name,
            "device_resolved": str(device),
            "trainable_latent_params": trainable_latent_params,
            "trainable_loss_coefficients": trainable_loss_coefficients,
            "total_optimized_params": trainable_latent_params + trainable_loss_coefficients,
            "target_param_count": target_param_count,
            "paper_table_1_acc": paper_acc,
            "optimizer_name": "AdamW",
        }
    )

    if args.protocol == "paper":
        config["paper_protocol_scope"] = "strict_for_disclosed_settings"
        config["paper_undisclosed_settings"] = list(PAPER_UNDISCLOSED_SETTINGS)

    if args.mapping_mode == "lwt":
        config["layerwise_dims"] = model.layer_dims
        config["layerwise_modulation_scales_resolved"] = (
            model.layer_modulation_scales
        )

    save_json(config, run_dir / "config.json")

    print("=" * 80)
    print("Paper 3.1.1 Mapping CNN reproduction")
    print("=" * 80)
    print(f"run_dir                  = {run_dir}")
    print(f"device                   = {device}")
    print(f"dataset                  = {args.dataset}")
    print(f"target model             = {args.model}")
    print(f"mapping_mode             = {args.mapping_mode}")
    print(f"projection_layout       = {args.projection_layout}")
    print(f"protocol                 = {args.protocol}")
    print(f"loss_mode                = {args.loss_mode}")
    if args.loss_mode == "full":
        print(f"loss_coefficient_mode    = {args.loss_coefficient_mode}")
        print(f"mapping_loss_terms       = {args.mapping_loss_terms}")
    print(f"trainable_latent_params  = {trainable_latent_params:,}")
    print(f"trainable_loss_coeffs    = {trainable_loss_coefficients}")
    print(f"total_optimized_params   = {trainable_latent_params + trainable_loss_coefficients:,}")
    print(f"target_generated_params  = {target_param_count:,}")
    print(f"paper_acc                = {paper_acc}")

    if args.mapping_mode == "lwt":
        print(f"layerwise_dims           = {model.layer_dims}")
        print(f"layerwise_alphas         = {model.layer_modulation_scales}")

    if args.protocol == "paper":
        print("paper_scope              = strict for explicitly disclosed settings")
        print("paper_unknowns           = activation, alpha/init values, latent allocation,")
        print("                           optimizer and training hyperparameters")

    print("=" * 80)

    best_test_acc = 0.0
    best_epoch = 0

    history_path = run_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_task_loss = 0.0
        total_stability_loss = 0.0
        total_smoothness_loss = 0.0
        total_alignment_loss = 0.0
        total_correct = 0
        total_count = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        )

        for batch_idx, (x, y) in enumerate(progress):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, parts, logits = compute_mapping_loss(
                args=args,
                model=model,
                loss_coefficients=loss_coefficients,
                x=x,
                y=y,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_task_loss += parts["task_loss"] * y.numel()
            total_stability_loss += parts["stability_loss"] * y.numel()
            total_smoothness_loss += parts["smoothness_loss"] * y.numel()
            total_alignment_loss += parts["alignment_loss"] * y.numel()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += y.numel()

            progress.set_postfix(
                loss=total_loss / total_count,
                task_loss=total_task_loss / total_count,
                acc=total_correct / total_count,
            )

        train_loss = total_loss / total_count
        train_task_loss = total_task_loss / total_count
        train_stability_loss = total_stability_loss / total_count
        train_smoothness_loss = total_smoothness_loss / total_count
        train_alignment_loss = total_alignment_loss / total_count
        train_acc = total_correct / total_count

        test_loss, test_acc = evaluate_mapping(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.max_test_batches,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_mapping_model.pt")
            if loss_coefficients is not None:
                torch.save(
                    loss_coefficients.state_dict(),
                    run_dir / "best_mapping_loss_coefficients.pt",
                )

        coefficient_values = (
            {"stability": 0.0, "smoothness": 0.0, "alignment": 0.0}
            if loss_coefficients is None
            else loss_coefficients.detached_values()
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_task_loss": train_task_loss,
            "train_stability_loss": train_stability_loss,
            "train_smoothness_loss": train_smoothness_loss,
            "train_alignment_loss": train_alignment_loss,
            "lambda_stability": coefficient_values["stability"],
            "lambda_smoothness": coefficient_values["smoothness"],
            "lambda_alignment": coefficient_values["alignment"],
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "paper_acc": paper_acc,
            "gap_best_minus_paper": None if paper_acc is None else best_test_acc - paper_acc,
            "trainable_latent_params": trainable_latent_params,
            "trainable_loss_coefficients": trainable_loss_coefficients,
            "target_param_count": target_param_count,
        }

        append_jsonl(row, history_path)

        paper_display = "n/a" if paper_acc is None else f"{paper_acc * 100:.2f}%"
        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_acc * 100:.2f}% "
            f"test_acc={test_acc * 100:.2f}% "
            f"best={best_test_acc * 100:.2f}% "
            f"paper={paper_display} "
            f"lambdas=({coefficient_values['stability']:.4g},"
            f"{coefficient_values['smoothness']:.4g},"
            f"{coefficient_values['alignment']:.4g})"
        )

    final_result = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "paper_acc": paper_acc,
        "gap_best_minus_paper": None if paper_acc is None else best_test_acc - paper_acc,
        "trainable_latent_params": trainable_latent_params,
        "trainable_loss_coefficients": trainable_loss_coefficients,
        "total_optimized_params": trainable_latent_params + trainable_loss_coefficients,
        "final_mapping_loss_coefficients": (
            None if loss_coefficients is None else loss_coefficients.detached_values()
        ),
        "target_param_count": target_param_count,
    }

    save_json(final_result, run_dir / "final_result.json")

    print("=" * 80)
    print("Finished Mapping run")
    print(f"best_epoch               = {best_epoch}")
    print(f"best_test_acc            = {best_test_acc * 100:.2f}%")
    if paper_acc is not None:
        print(f"paper_acc                = {paper_acc * 100:.2f}%")
        print(f"gap                      = {(best_test_acc - paper_acc) * 100:.2f}%")
    print(f"saved to                 = {run_dir}")
    print("=" * 80)


# ============================================================
# 8. 汇总结果
# ============================================================

def summarize_results(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)

    final_files = sorted(results_dir.glob("*/final_result.json"))

    if not final_files:
        print(f"No final_result.json found under {results_dir}")
        return

    rows = []

    for file in final_files:
        obj = json.loads(file.read_text(encoding="utf-8"))

        rows.append(
            {
                "run": file.parent.name,
                "best_acc": obj.get("best_test_acc"),
                "paper_acc": obj.get("paper_acc"),
                "gap": obj.get("gap_best_minus_paper"),
                "trainable": obj.get("trainable_latent_params"),
                "target": obj.get("target_param_count"),
            }
        )

    print("=" * 120)
    print("Summary")
    print("=" * 120)

    for row in rows:
        best = row["best_acc"]
        paper = row["paper_acc"]
        gap = row["gap"]

        best_str = "None" if best is None else f"{best * 100:.2f}%"
        paper_str = "None" if paper is None else f"{paper * 100:.2f}%"
        gap_str = "None" if gap is None else f"{gap * 100:+.2f}%"

        print(
            f"{row['run']:<70} "
            f"best={best_str:<8} "
            f"paper={paper_str:<8} "
            f"gap={gap_str:<8} "
            f"trainable={row['trainable']} "
            f"target={row['target']}"
        )

    print("=" * 120)


# ============================================================
# 9. 命令行参数
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper 3.1.1 reproduction: CNN1/CNN2 and Mapping CNN on MNIST/FashionMNIST"
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "mapping", "summary"],
        required=True,
        help="baseline: train CNN1/CNN2; mapping: train Ours*/Ours†; summary: summarize result folders",
    )

    parser.add_argument(
        "--protocol",
        choices=["paper", "custom"],
        default="paper",
        help=(
            "paper enforces all explicitly disclosed Mapping Network choices; "
            "custom permits exploratory alternatives"
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=["mnist", "fmnist"],
        default="mnist",
    )

    parser.add_argument(
        "--model",
        choices=["cnn1", "cnn2"],
        default="cnn2",
    )

    parser.add_argument(
        "--mapping-mode",
        choices=["slvt", "lwt"],
        default="slvt",
        help="slvt = Ours*, lwt = Ours†",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="SLVT / Ours* latent dimension",
    )

    parser.add_argument(
        "--total-latent-dim",
        type=int,
        default=None,
        help="LWT / Ours† total latent dimension",
    )

    parser.add_argument(
        "--layerwise-latent-dims",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit LWT dimensions in model layer order",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
    )

    parser.add_argument(
        "--results-dir",
        default="results",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
    )

    # Mapping-specific options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of generated parameters per fixed projection chunk",
    )

    parser.add_argument(
        "--activation",
        choices=["tanh", "identity", "sin"],
        default="tanh",
    )

    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--projection-init",
        choices=["orthogonal", "gaussian"],
        default="orthogonal",
        help="Paper mode uses fixed orthogonally initialized mapping weights",
    )

    parser.add_argument(
        "--projection-layout",
        choices=["global", "blockwise"],
        default="global",
        help=(
            "global initializes one orthogonal mapping matrix; blockwise "
            "reproduces the legacy independently initialized chunks"
        ),
    )

    parser.add_argument(
        "--modulation-reduction",
        choices=["sum", "mean"],
        default="sum",
        help="Paper equation expands to alpha * sum(z^2); mean preserves legacy runs",
    )

    parser.add_argument(
        "--parameter-scale-mode",
        choices=["paper", "fan-in"],
        default="paper",
        help="paper uses mapped parameters directly; fan-in preserves legacy scaling",
    )

    parser.add_argument(
        "--modulation-scale",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--layerwise-modulation-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional LWT modulation scales in model layer order; "
            "overrides --modulation-scale for each layer"
        ),
    )

    parser.add_argument(
        "--latent-init-std",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--loss-mode",
        choices=["task", "full"],
        default="task",
        help="task uses cross-entropy only; full implements all paper Mapping Loss terms",
    )

    parser.add_argument(
        "--loss-coefficient-mode",
        choices=["trainable", "fixed"],
        default="trainable",
        help="Train Mapping Loss coefficients or keep their CLI values fixed",
    )

    parser.add_argument(
        "--mapping-loss-terms",
        nargs="+",
        choices=list(MAPPING_LOSS_TERMS),
        default=list(MAPPING_LOSS_TERMS),
        help="Auxiliary Mapping Loss terms enabled when --loss-mode full",
    )

    parser.add_argument(
        "--lambda-stability",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--lambda-smoothness",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lambda-alignment",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--stability-sigma",
        type=float,
        default=0.01,
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "baseline":
        train_baseline(args)

    elif args.mode == "mapping":
        train_mapping(args)

    elif args.mode == "summary":
        summarize_results(args)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
