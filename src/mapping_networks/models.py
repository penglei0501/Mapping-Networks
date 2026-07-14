from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LayerSpec:
    in_features: int
    out_features: int

    @property
    def weight_numel(self) -> int:
        return self.in_features * self.out_features

    @property
    def bias_numel(self) -> int:
        return self.out_features

    @property
    def numel(self) -> int:
        return self.weight_numel + self.bias_numel


class DirectMLP(nn.Module):
    """Plain MLP baseline with directly trained weights."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Iterable[int] = (256, 256),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class DirectCNN(nn.Module):
    """Small CNN baseline for MNIST/FashionMNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DirectCNN1(nn.Module):
    """CNN1 from the paper supplement: AlexNet-like classifier for 28x28 digits."""

    def __init__(self, num_classes: int = 10) -> None:
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
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DirectCNN2(nn.Module):
    """CNN2 from the paper supplement: LeNet-like classifier for 28x28 digits."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EfficientCNN(nn.Module):
    """Compact CNN with most capacity in convolutional features, not a large FC head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(8, 96)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.gn1(self.conv1(x)))
        x = F.silu(self.gn2(self.conv2(x)))
        x = F.silu(self.gn3(self.conv3(x)))
        x = F.silu(self.gn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class FourierFeatures(nn.Module):
    """Deterministic coordinate features for the mapping network."""

    def __init__(self, num_bands: int = 8) -> None:
        super().__init__()
        bands = torch.tensor([2.0**i for i in range(num_bands)], dtype=torch.float32)
        self.register_buffer("bands", bands)

    @property
    def out_dim(self) -> int:
        return 2 + 4 * int(self.bands.numel())

    def forward(self, layer_coord: Tensor, param_coord: Tensor) -> Tensor:
        coords = torch.stack((layer_coord, param_coord), dim=-1)
        angles = coords[..., None] * self.bands * torch.pi
        encoded = torch.cat((coords, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)), dim=-1)
        return encoded


class MappingMLP(nn.Module):
    """MLP whose weights are generated from a trainable latent vector.

    The target classifier parameters are not learned directly. Instead, a
    shared mapping network receives a layer coordinate, parameter coordinate,
    and the trainable latent vector, then emits every scalar target parameter.
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Iterable[int] = (256, 256),
        num_classes: int = 10,
        latent_dim: int = 64,
        mapper_width: int = 128,
        mapper_depth: int = 3,
        fourier_bands: int = 8,
        train_mapper: bool = False,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        self.layer_specs = [LayerSpec(a, b) for a, b in zip(dims, dims[1:])]
        self.latent = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.features = FourierFeatures(fourier_bands)

        mapper_layers: list[nn.Module] = []
        mapper_input_dim = latent_dim + self.features.out_dim
        for layer_index in range(mapper_depth):
            mapper_layers.append(
                nn.Linear(mapper_input_dim if layer_index == 0 else mapper_width, mapper_width)
            )
            mapper_layers.append(nn.SiLU())
        mapper_layers.append(nn.Linear(mapper_width, 1))
        self.mapper = nn.Sequential(*mapper_layers)
        self.reset_mapper()

        if not train_mapper:
            for parameter in self.mapper.parameters():
                parameter.requires_grad_(False)

    @property
    def target_parameter_count(self) -> int:
        return sum(spec.numel for spec in self.layer_specs)

    @property
    def trained_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def reset_mapper(self) -> None:
        for module in self.mapper.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.mapper[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=0.02)
            nn.init.zeros_(final.bias)

    def _generated_parameters(self, device: torch.device) -> list[tuple[Tensor, Tensor]]:
        generated: list[tuple[Tensor, Tensor]] = []
        layer_count = max(1, len(self.layer_specs) - 1)

        for layer_index, spec in enumerate(self.layer_specs):
            numel = spec.numel
            layer_coord = torch.full((numel,), layer_index / layer_count, device=device)
            param_coord = torch.linspace(-1.0, 1.0, steps=numel, device=device)
            features = self.features(layer_coord, param_coord)
            latent = self.latent.to(device).expand(numel, -1)
            raw = self.mapper(torch.cat((features, latent), dim=-1)).squeeze(-1)

            scale = 1.0 / (spec.in_features**0.5)
            raw = raw * scale
            weight = raw[: spec.weight_numel].view(spec.out_features, spec.in_features)
            bias = raw[spec.weight_numel :].view(spec.out_features)
            generated.append((weight, bias))

        return generated

    def forward(self, x: Tensor) -> Tensor:
        activations = x.flatten(1)
        params = self._generated_parameters(activations.device)

        for layer_index, (weight, bias) in enumerate(params):
            activations = F.linear(activations, weight, bias)
            if layer_index < len(params) - 1:
                activations = F.relu(activations)
        return activations


class ProjectionMappingMLP(nn.Module):
    """Paper-style fixed projection mapper with latent weight modulation."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Iterable[int] = (256, 256),
        num_classes: int = 10,
        latent_dim: int = 1024,
        modulation_scale: float = 0.01,
        activation: Literal["tanh", "identity"] = "tanh",
        layerwise: bool = False,
        projection_gain: float = 1.0,
        latent_init_std: float = 1.0,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        self.layer_specs = [LayerSpec(a, b) for a, b in zip(dims, dims[1:])]
        self.modulation_scale = modulation_scale
        self.activation = activation
        self.layerwise = layerwise
        self.projection_gain = projection_gain
        self.output_gain = output_gain

        if layerwise:
            self.latents = nn.ParameterList(
                [nn.Parameter(torch.randn(latent_dim) * latent_init_std) for _ in self.layer_specs]
            )
            for layer_index, spec in enumerate(self.layer_specs):
                projection = torch.empty(spec.numel, latent_dim)
                nn.init.orthogonal_(projection)
                projection.mul_(projection_gain)
                bias = torch.zeros(spec.numel)
                self.register_buffer(f"projection_{layer_index}", projection)
                self.register_buffer(f"bias_{layer_index}", bias)
        else:
            self.latent = nn.Parameter(torch.randn(latent_dim) * latent_init_std)
            projection = torch.empty(self.target_parameter_count, latent_dim)
            nn.init.orthogonal_(projection)
            projection.mul_(projection_gain)
            self.register_buffer("projection", projection)
            self.register_buffer("bias", torch.zeros(self.target_parameter_count))

    @property
    def target_parameter_count(self) -> int:
        return sum(spec.numel for spec in self.layer_specs)

    @property
    def trained_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _project(self, projection: Tensor, bias: Tensor, latent: Tensor) -> Tensor:
        # Implements W_ij <- W_ij + alpha z_i without materializing the modulated matrix.
        raw = projection @ latent + self.modulation_scale * torch.mean(latent * latent) + bias
        if self.activation == "tanh":
            return torch.tanh(raw)
        return raw

    def _generated_flat_parameters(self) -> Tensor:
        if not self.layerwise:
            return self._project(self.projection, self.bias, self.latent)

        chunks = []
        for layer_index, latent in enumerate(self.latents):
            projection = getattr(self, f"projection_{layer_index}")
            bias = getattr(self, f"bias_{layer_index}")
            chunks.append(self._project(projection, bias, latent))
        return torch.cat(chunks)

    def _generated_parameters(self) -> list[tuple[Tensor, Tensor]]:
        flat = self._generated_flat_parameters()
        generated: list[tuple[Tensor, Tensor]] = []
        offset = 0
        for spec in self.layer_specs:
            layer = flat[offset : offset + spec.numel]
            offset += spec.numel
            scale = 1.0 / (spec.in_features**0.5)
            weight = layer[: spec.weight_numel].view(spec.out_features, spec.in_features) * scale * self.output_gain
            bias = layer[spec.weight_numel :].view(spec.out_features) * scale * self.output_gain
            generated.append((weight, bias))
        return generated

    def forward(self, x: Tensor) -> Tensor:
        activations = x.flatten(1)
        params = self._generated_parameters()

        for layer_index, (weight, bias) in enumerate(params):
            activations = F.linear(activations, weight, bias)
            if layer_index < len(params) - 1:
                activations = F.relu(activations)
        return activations


class ProjectionMappingCNN(nn.Module):
    """Small CNN whose parameters are generated by fixed projection mappers."""

    def __init__(
        self,
        latent_dim: int = 512,
        modulation_scale: float = 0.01,
        activation: Literal["tanh", "identity"] = "identity",
        layerwise: bool = True,
        projection_gain: float = 1.0,
        latent_init_std: float = 1.0,
        output_gain: float = 1.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.param_specs: list[tuple[str, tuple[int, ...]]] = [
            ("conv1.weight", (16, 1, 3, 3)),
            ("conv1.bias", (16,)),
            ("conv2.weight", (32, 16, 3, 3)),
            ("conv2.bias", (32,)),
            ("fc1.weight", (64, 32 * 7 * 7)),
            ("fc1.bias", (64,)),
            ("fc2.weight", (num_classes, 64)),
            ("fc2.bias", (num_classes,)),
        ]
        self.groups = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
        self.modulation_scale = modulation_scale
        self.activation = activation
        self.layerwise = layerwise
        self.output_gain = output_gain

        if layerwise:
            self.latents = nn.ParameterList(
                [nn.Parameter(torch.randn(latent_dim) * latent_init_std) for _ in self.groups]
            )
            for group_index, group in enumerate(self.groups):
                group_numel = sum(prod(self.param_specs[index][1]) for index in group)
                projection = torch.empty(group_numel, latent_dim)
                nn.init.orthogonal_(projection)
                projection.mul_(projection_gain)
                self.register_buffer(f"projection_{group_index}", projection)
                self.register_buffer(f"bias_{group_index}", torch.zeros(group_numel))
        else:
            self.latent = nn.Parameter(torch.randn(latent_dim) * latent_init_std)
            projection = torch.empty(self.target_parameter_count, latent_dim)
            nn.init.orthogonal_(projection)
            projection.mul_(projection_gain)
            self.register_buffer("projection", projection)
            self.register_buffer("bias", torch.zeros(self.target_parameter_count))

    @property
    def target_parameter_count(self) -> int:
        return sum(prod(shape) for _, shape in self.param_specs)

    @property
    def trained_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _project(self, projection: Tensor, bias: Tensor, latent: Tensor) -> Tensor:
        raw = projection @ latent + self.modulation_scale * torch.sum(latent * latent) + bias
        if self.activation == "tanh":
            return torch.tanh(raw)
        return raw

    def _latent_projection_biases(self) -> list[tuple[Tensor, Tensor, Tensor]]:
        if not self.layerwise:
            return [(self.projection, self.bias, self.latent)]

        entries = []
        for group_index, latent in enumerate(self.latents):
            projection = getattr(self, f"projection_{group_index}")
            bias = getattr(self, f"bias_{group_index}")
            entries.append((projection, bias, latent))
        return entries

    def _generated_flat_parameters(self, latents: Iterable[Tensor] | Tensor | None = None) -> Tensor:
        if not self.layerwise:
            latent = self.latent if latents is None else latents
            if not isinstance(latent, Tensor):
                latent = list(latent)[0]
            return self._project(self.projection, self.bias, latent)

        latent_list = list(self.latents if latents is None else latents)
        if len(latent_list) != len(self.groups):
            raise ValueError(f"Expected {len(self.groups)} latent tensors, got {len(latent_list)}.")

        chunks = []
        for group_index, latent in enumerate(latent_list):
            projection = getattr(self, f"projection_{group_index}")
            bias = getattr(self, f"bias_{group_index}")
            chunks.append(self._project(projection, bias, latent))
        return torch.cat(chunks)

    def _generated_parameters(self, latents: Iterable[Tensor] | Tensor | None = None) -> dict[str, Tensor]:
        flat = self._generated_flat_parameters(latents)
        parameters: dict[str, Tensor] = {}
        offset = 0
        for name, shape in self.param_specs:
            numel = prod(shape)
            value = flat[offset : offset + numel].view(shape)
            offset += numel

            if name.endswith("weight"):
                fan_in = prod(shape[1:])
                value = value * (self.output_gain / fan_in**0.5)
            else:
                value = value * self.output_gain * 0.01
            parameters[name] = value
        return parameters

    def mapping_loss_terms(
        self,
        x: Tensor,
        *,
        clean_logits: Tensor | None = None,
        perturb_std: float = 1e-3,
    ) -> dict[str, Tensor]:
        if perturb_std <= 0:
            raise ValueError("perturb_std must be positive.")

        clean_logits = self(x) if clean_logits is None else clean_logits
        current_latents = [latent for _, _, latent in self._latent_projection_biases()]
        perturbations = [torch.randn_like(latent) * perturb_std for latent in current_latents]
        perturbed_latents = [latent + perturb for latent, perturb in zip(current_latents, perturbations)]

        perturbed_logits = self.forward_with_latents(x, perturbed_latents if self.layerwise else perturbed_latents[0])
        stability = F.mse_loss(perturbed_logits, clean_logits)

        clean_flat = self._generated_flat_parameters()
        perturbed_flat = self._generated_flat_parameters(perturbed_latents if self.layerwise else perturbed_latents[0])
        smoothness = F.mse_loss(perturbed_flat, clean_flat) / (perturb_std**2)

        alignment_terms = []
        for projection, _, latent in self._latent_projection_biases():
            dominant_direction = projection.mean(dim=0) + self.modulation_scale * latent
            cosine = F.cosine_similarity(latent.flatten(), dominant_direction.flatten(), dim=0).clamp(-1.0, 1.0)
            alignment_terms.append(1.0 - cosine)
        alignment = torch.stack(alignment_terms).mean()

        return {
            "stability": stability,
            "smoothness": smoothness,
            "alignment": alignment,
        }

    def forward_with_latents(self, x: Tensor, latents: Iterable[Tensor] | Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        params = self._generated_parameters()
        return self._forward_with_parameters(x, params)

    def forward_with_latents(self, x: Tensor, latents: Iterable[Tensor] | Tensor) -> Tensor:
        params = self._generated_parameters(latents)
        return self._forward_with_parameters(x, params)

    def _forward_with_parameters(self, x: Tensor, params: dict[str, Tensor]) -> Tensor:
        x = F.conv2d(x, params["conv1.weight"], params["conv1.bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, params["conv2.weight"], params["conv2.bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        return F.linear(x, params["fc2.weight"], params["fc2.bias"])


class _ProjectionMappingFunctionalCNN(nn.Module):
    """Shared fixed-projection parameter generator for functional CNN variants."""

    param_specs: list[tuple[str, tuple[int, ...]]]
    groups: list[list[int]]

    def _init_projection_mapping(
        self,
        *,
        latent_dim: int,
        modulation_scale: float,
        activation: Literal["tanh", "identity"],
        layerwise: bool,
        layerwise_latent_dims: Iterable[int] | None,
        projection_gain: float,
        latent_init_std: float,
        output_gain: float,
        parameter_scale_mode: Literal["fan-in", "paper"],
    ) -> None:
        self.modulation_scale = modulation_scale
        self.activation = activation
        self.layerwise = layerwise
        self.output_gain = output_gain
        self.parameter_scale_mode = parameter_scale_mode

        if layerwise:
            latent_dims = (
                list(layerwise_latent_dims)
                if layerwise_latent_dims is not None
                else [latent_dim for _ in self.groups]
            )
            if len(latent_dims) != len(self.groups):
                raise ValueError(
                    f"Expected {len(self.groups)} layerwise latent dims, got {len(latent_dims)}."
                )
            if any(dim <= 0 for dim in latent_dims):
                raise ValueError("Layerwise latent dims must be positive.")

            self.latents = nn.ParameterList(
                [nn.Parameter(torch.randn(group_latent_dim) * latent_init_std) for group_latent_dim in latent_dims]
            )
            for group_index, group in enumerate(self.groups):
                group_numel = sum(prod(self.param_specs[index][1]) for index in group)
                projection = torch.empty(group_numel, latent_dims[group_index])
                nn.init.orthogonal_(projection)
                projection.mul_(projection_gain)
                self.register_buffer(f"projection_{group_index}", projection)
                self.register_buffer(f"bias_{group_index}", torch.zeros(group_numel))
        else:
            self.latent = nn.Parameter(torch.randn(latent_dim) * latent_init_std)
            projection = torch.empty(self.target_parameter_count, latent_dim)
            nn.init.orthogonal_(projection)
            projection.mul_(projection_gain)
            self.register_buffer("projection", projection)
            self.register_buffer("bias", torch.zeros(self.target_parameter_count))

    @property
    def target_parameter_count(self) -> int:
        return sum(prod(shape) for _, shape in self.param_specs)

    @property
    def trained_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _project(self, projection: Tensor, bias: Tensor, latent: Tensor) -> Tensor:
        raw = projection @ latent + self.modulation_scale * torch.sum(latent * latent) + bias
        if self.activation == "tanh":
            return torch.tanh(raw)
        return raw

    def _latent_projection_biases(self) -> list[tuple[Tensor, Tensor, Tensor]]:
        if not self.layerwise:
            return [(self.projection, self.bias, self.latent)]

        entries = []
        for group_index, latent in enumerate(self.latents):
            projection = getattr(self, f"projection_{group_index}")
            bias = getattr(self, f"bias_{group_index}")
            entries.append((projection, bias, latent))
        return entries

    def _generated_flat_parameters(self, latents: Iterable[Tensor] | Tensor | None = None) -> Tensor:
        if not self.layerwise:
            latent = self.latent if latents is None else latents
            if not isinstance(latent, Tensor):
                latent = list(latent)[0]
            return self._project(self.projection, self.bias, latent)

        latent_list = list(self.latents if latents is None else latents)
        if len(latent_list) != len(self.groups):
            raise ValueError(f"Expected {len(self.groups)} latent tensors, got {len(latent_list)}.")

        chunks = []
        for group_index, latent in enumerate(latent_list):
            projection = getattr(self, f"projection_{group_index}")
            bias = getattr(self, f"bias_{group_index}")
            chunks.append(self._project(projection, bias, latent))
        return torch.cat(chunks)

    def _generated_parameters(self, latents: Iterable[Tensor] | Tensor | None = None) -> dict[str, Tensor]:
        flat = self._generated_flat_parameters(latents)
        parameters: dict[str, Tensor] = {}
        offset = 0
        for name, shape in self.param_specs:
            numel = prod(shape)
            value = flat[offset : offset + numel].view(shape)
            offset += numel

            if self.parameter_scale_mode == "paper":
                value = value * self.output_gain
            elif name.endswith("weight"):
                fan_in = prod(shape[1:])
                value = value * (self.output_gain / fan_in**0.5)
            else:
                value = value * self.output_gain * 0.01
            parameters[name] = value
        return parameters

    def mapping_loss_terms(
        self,
        x: Tensor,
        *,
        clean_logits: Tensor | None = None,
        perturb_std: float = 1e-3,
    ) -> dict[str, Tensor]:
        if perturb_std <= 0:
            raise ValueError("perturb_std must be positive.")

        clean_logits = self(x) if clean_logits is None else clean_logits
        current_latents = [latent for _, _, latent in self._latent_projection_biases()]
        perturbations = [torch.randn_like(latent) * perturb_std for latent in current_latents]
        perturbed_latents = [latent + perturb for latent, perturb in zip(current_latents, perturbations)]
        latent_override = perturbed_latents if self.layerwise else perturbed_latents[0]

        perturbed_logits = self.forward_with_latents(x, latent_override)
        stability = F.mse_loss(perturbed_logits, clean_logits)

        clean_flat = self._generated_flat_parameters()
        perturbed_flat = self._generated_flat_parameters(latent_override)
        smoothness = F.mse_loss(perturbed_flat, clean_flat) / (perturb_std**2)

        alignment_terms = []
        for projection, _, latent in self._latent_projection_biases():
            dominant_direction = projection.mean(dim=0) + self.modulation_scale * latent
            cosine = F.cosine_similarity(latent.flatten(), dominant_direction.flatten(), dim=0).clamp(-1.0, 1.0)
            alignment_terms.append(1.0 - cosine)
        alignment = torch.stack(alignment_terms).mean()

        return {
            "stability": stability,
            "smoothness": smoothness,
            "alignment": alignment,
        }

    def forward_with_latents(self, x: Tensor, latents: Iterable[Tensor] | Tensor) -> Tensor:
        raise NotImplementedError


class ProjectionMappingCNN1(_ProjectionMappingFunctionalCNN):
    """Projection-mapped CNN1 matching the paper supplement architecture."""

    def __init__(
        self,
        latent_dim: int = 512,
        modulation_scale: float = 0.01,
        activation: Literal["tanh", "identity"] = "identity",
        layerwise: bool = True,
        layerwise_latent_dims: Iterable[int] | None = None,
        projection_gain: float = 1.0,
        latent_init_std: float = 1.0,
        output_gain: float = 1.0,
        parameter_scale_mode: Literal["fan-in", "paper"] = "fan-in",
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.param_specs = [
            ("conv1.weight", (32, 1, 3, 3)),
            ("conv1.bias", (32,)),
            ("conv2.weight", (64, 32, 3, 3)),
            ("conv2.bias", (64,)),
            ("conv3.weight", (128, 64, 3, 3)),
            ("conv3.bias", (128,)),
            ("conv4.weight", (128, 128, 3, 3)),
            ("conv4.bias", (128,)),
            ("fc1.weight", (256, 128 * 3 * 3)),
            ("fc1.bias", (256,)),
            ("fc2.weight", (num_classes, 256)),
            ("fc2.bias", (num_classes,)),
        ]
        self.groups = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
        ]
        self._init_projection_mapping(
            latent_dim=latent_dim,
            modulation_scale=modulation_scale,
            activation=activation,
            layerwise=layerwise,
            layerwise_latent_dims=layerwise_latent_dims,
            projection_gain=projection_gain,
            latent_init_std=latent_init_std,
            output_gain=output_gain,
            parameter_scale_mode=parameter_scale_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        params = self._generated_parameters()
        return self._forward_with_parameters(x, params)

    def forward_with_latents(self, x: Tensor, latents: Iterable[Tensor] | Tensor) -> Tensor:
        params = self._generated_parameters(latents)
        return self._forward_with_parameters(x, params)

    def _forward_with_parameters(self, x: Tensor, params: dict[str, Tensor]) -> Tensor:
        x = F.conv2d(x, params["conv1.weight"], params["conv1.bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, params["conv2.weight"], params["conv2.bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, params["conv3.weight"], params["conv3.bias"], padding=1)
        x = F.relu(x)
        x = F.conv2d(x, params["conv4.weight"], params["conv4.bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        return F.linear(x, params["fc2.weight"], params["fc2.bias"])


class ProjectionMappingCNN2(_ProjectionMappingFunctionalCNN):
    """Projection-mapped CNN2 matching the paper supplement architecture."""

    def __init__(
        self,
        latent_dim: int = 512,
        modulation_scale: float = 0.01,
        activation: Literal["tanh", "identity"] = "identity",
        layerwise: bool = True,
        layerwise_latent_dims: Iterable[int] | None = None,
        projection_gain: float = 1.0,
        latent_init_std: float = 1.0,
        output_gain: float = 1.0,
        parameter_scale_mode: Literal["fan-in", "paper"] = "fan-in",
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.param_specs = [
            ("conv1.weight", (16, 1, 3, 3)),
            ("conv1.bias", (16,)),
            ("conv2.weight", (32, 16, 3, 3)),
            ("conv2.bias", (32,)),
            ("fc1.weight", (128, 32 * 5 * 5)),
            ("fc1.bias", (128,)),
            ("fc2.weight", (num_classes, 128)),
            ("fc2.bias", (num_classes,)),
        ]
        self.groups = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
        self._init_projection_mapping(
            latent_dim=latent_dim,
            modulation_scale=modulation_scale,
            activation=activation,
            layerwise=layerwise,
            layerwise_latent_dims=layerwise_latent_dims,
            projection_gain=projection_gain,
            latent_init_std=latent_init_std,
            output_gain=output_gain,
            parameter_scale_mode=parameter_scale_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        params = self._generated_parameters()
        return self._forward_with_parameters(x, params)

    def forward_with_latents(self, x: Tensor, latents: Iterable[Tensor] | Tensor) -> Tensor:
        params = self._generated_parameters(latents)
        return self._forward_with_parameters(x, params)

    def _forward_with_parameters(self, x: Tensor, params: dict[str, Tensor]) -> Tensor:
        x = F.conv2d(x, params["conv1.weight"], params["conv1.bias"])
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, params["conv2.weight"], params["conv2.bias"])
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        return F.linear(x, params["fc2.weight"], params["fc2.bias"])


class ProjectionMappingEfficientCNN(nn.Module):
    """Projection-mapped version of EfficientCNN."""

    def __init__(
        self,
        latent_dim: int = 512,
        modulation_scale: float = 0.01,
        activation: Literal["tanh", "identity"] = "identity",
        layerwise: bool = True,
        projection_gain: float = 1.0,
        latent_init_std: float = 1.0,
        output_gain: float = 1.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.param_specs: list[tuple[str, tuple[int, ...]]] = [
            ("conv1.weight", (24, 1, 3, 3)),
            ("conv1.bias", (24,)),
            ("gn1.weight", (24,)),
            ("gn1.bias", (24,)),
            ("conv2.weight", (48, 24, 3, 3)),
            ("conv2.bias", (48,)),
            ("gn2.weight", (48,)),
            ("gn2.bias", (48,)),
            ("conv3.weight", (64, 48, 3, 3)),
            ("conv3.bias", (64,)),
            ("gn3.weight", (64,)),
            ("gn3.bias", (64,)),
            ("conv4.weight", (96, 64, 3, 3)),
            ("conv4.bias", (96,)),
            ("gn4.weight", (96,)),
            ("gn4.bias", (96,)),
            ("fc.weight", (num_classes, 96)),
            ("fc.bias", (num_classes,)),
        ]
        self.groups = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17],
        ]
        self.modulation_scale = modulation_scale
        self.activation = activation
        self.layerwise = layerwise
        self.output_gain = output_gain

        if layerwise:
            self.latents = nn.ParameterList(
                [nn.Parameter(torch.randn(latent_dim) * latent_init_std) for _ in self.groups]
            )
            for group_index, group in enumerate(self.groups):
                group_numel = sum(prod(self.param_specs[index][1]) for index in group)
                projection = torch.empty(group_numel, latent_dim)
                nn.init.orthogonal_(projection)
                projection.mul_(projection_gain)
                self.register_buffer(f"projection_{group_index}", projection)
                self.register_buffer(f"bias_{group_index}", torch.zeros(group_numel))
        else:
            self.latent = nn.Parameter(torch.randn(latent_dim) * latent_init_std)
            projection = torch.empty(self.target_parameter_count, latent_dim)
            nn.init.orthogonal_(projection)
            projection.mul_(projection_gain)
            self.register_buffer("projection", projection)
            self.register_buffer("bias", torch.zeros(self.target_parameter_count))

    @property
    def target_parameter_count(self) -> int:
        return sum(prod(shape) for _, shape in self.param_specs)

    @property
    def trained_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _project(self, projection: Tensor, bias: Tensor, latent: Tensor) -> Tensor:
        raw = projection @ latent + self.modulation_scale * torch.mean(latent * latent) + bias
        if self.activation == "tanh":
            return torch.tanh(raw)
        return raw

    def _generated_flat_parameters(self) -> Tensor:
        if not self.layerwise:
            return self._project(self.projection, self.bias, self.latent)

        chunks = []
        for group_index, latent in enumerate(self.latents):
            projection = getattr(self, f"projection_{group_index}")
            bias = getattr(self, f"bias_{group_index}")
            chunks.append(self._project(projection, bias, latent))
        return torch.cat(chunks)

    def _generated_parameters(self) -> dict[str, Tensor]:
        flat = self._generated_flat_parameters()
        parameters: dict[str, Tensor] = {}
        offset = 0
        for name, shape in self.param_specs:
            numel = prod(shape)
            value = flat[offset : offset + numel].view(shape)
            offset += numel

            if name.endswith("weight") and name.startswith(("conv", "fc")):
                fan_in = prod(shape[1:])
                value = value * (self.output_gain / fan_in**0.5)
            elif name.endswith("bias") and name.startswith(("conv", "fc")):
                value = value * self.output_gain * 0.01
            elif name.endswith("weight"):
                value = 1.0 + value * 0.05
            else:
                value = value * 0.05
            parameters[name] = value
        return parameters

    def forward(self, x: Tensor) -> Tensor:
        params = self._generated_parameters()
        x = F.conv2d(x, params["conv1.weight"], params["conv1.bias"], padding=1)
        x = F.group_norm(x, num_groups=4, weight=params["gn1.weight"], bias=params["gn1.bias"])
        x = F.silu(x)
        x = F.conv2d(x, params["conv2.weight"], params["conv2.bias"], stride=2, padding=1)
        x = F.group_norm(x, num_groups=8, weight=params["gn2.weight"], bias=params["gn2.bias"])
        x = F.silu(x)
        x = F.conv2d(x, params["conv3.weight"], params["conv3.bias"], stride=2, padding=1)
        x = F.group_norm(x, num_groups=8, weight=params["gn3.weight"], bias=params["gn3.bias"])
        x = F.silu(x)
        x = F.conv2d(x, params["conv4.weight"], params["conv4.bias"], padding=1)
        x = F.group_norm(x, num_groups=8, weight=params["gn4.weight"], bias=params["gn4.bias"])
        x = F.silu(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return F.linear(x, params["fc.weight"], params["fc.bias"])


def count_parameters(module: nn.Module) -> int:
    return sum(prod(parameter.shape) for parameter in module.parameters())
