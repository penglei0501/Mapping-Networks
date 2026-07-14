import argparse
import unittest

import torch

from experiments.train_digits import MappingLossCoefficients, build_model, compute_loss
from mapping_networks.models import (
    DirectCNN1,
    DirectCNN2,
    ProjectionMappingCNN1,
    ProjectionMappingCNN2,
    count_parameters,
)


class PaperClassificationModelTests(unittest.TestCase):
    def test_direct_cnn_parameter_counts_match_supplement(self) -> None:
        self.assertEqual(count_parameters(DirectCNN1()), 537_994)
        self.assertEqual(count_parameters(DirectCNN2()), 108_618)

    def test_projection_targets_match_direct_architectures(self) -> None:
        self.assertEqual(ProjectionMappingCNN1(latent_dim=32).target_parameter_count, 537_994)
        self.assertEqual(ProjectionMappingCNN2(latent_dim=32).target_parameter_count, 108_618)

    def test_projection_models_return_digit_logits(self) -> None:
        images = torch.randn(2, 1, 28, 28)
        self.assertEqual(ProjectionMappingCNN1(latent_dim=8)(images).shape, (2, 10))
        self.assertEqual(ProjectionMappingCNN2(latent_dim=8)(images).shape, (2, 10))

    def test_layerwise_projection_trains_one_latent_per_layer_group(self) -> None:
        cnn1 = ProjectionMappingCNN1(latent_dim=16, layerwise=True)
        cnn2 = ProjectionMappingCNN2(latent_dim=16, layerwise=True)

        self.assertEqual(cnn1.trained_parameter_count, 6 * 16)
        self.assertEqual(cnn2.trained_parameter_count, 4 * 16)

    def test_layerwise_projection_accepts_custom_latent_dims(self) -> None:
        cnn1 = ProjectionMappingCNN1(layerwise=True, layerwise_latent_dims=[2067, 1, 1, 1, 1, 1])
        cnn2 = ProjectionMappingCNN2(layerwise=True, layerwise_latent_dims=[512, 512, 512, 512])

        self.assertEqual(cnn1.trained_parameter_count, 2_072)
        self.assertEqual(cnn2.trained_parameter_count, 2_048)

    def test_projection_models_expose_paper_mapping_loss_terms(self) -> None:
        model = ProjectionMappingCNN2(latent_dim=8)
        images = torch.randn(2, 1, 28, 28)
        logits = model(images)

        terms = model.mapping_loss_terms(images, clean_logits=logits, perturb_std=1e-3)

        self.assertEqual(set(terms), {"stability", "smoothness", "alignment"})
        for value in terms.values():
            self.assertEqual(value.ndim, 0)
            self.assertGreaterEqual(float(value.detach()), 0.0)

    def test_projection_modulation_matches_paper_formula(self) -> None:
        model = ProjectionMappingCNN2(latent_dim=2, activation="identity", modulation_scale=0.1)
        projection = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        bias = torch.tensor([0.5, -0.5])
        latent = torch.tensor([2.0, -1.0])

        projected = model._project(projection, bias, latent)

        expected = projection @ latent + 0.1 * torch.sum(latent * latent) + bias
        torch.testing.assert_close(projected, expected)

    def test_paper_parameter_scale_mode_does_not_apply_fan_in_scaling(self) -> None:
        model = ProjectionMappingCNN2(
            latent_dim=2,
            activation="identity",
            modulation_scale=0.0,
            output_gain=1.0,
            parameter_scale_mode="paper",
        )
        flat = model._generated_flat_parameters()
        params = model._generated_parameters()

        conv1_numel = params["conv1.weight"].numel()
        torch.testing.assert_close(params["conv1.weight"].flatten(), flat[:conv1_numel])

    def test_train_script_passes_parameter_scale_mode_to_projection_cnn2(self) -> None:
        args = argparse.Namespace(
            model="projection-cnn2",
            latent_dim=2,
            modulation_scale=0.0,
            activation="identity",
            layerwise=False,
            layerwise_latent_dims=None,
            projection_gain=1.0,
            latent_init_std=1.0,
            output_gain=1.0,
            parameter_scale_mode="paper",
        )

        model = build_model(args)

        self.assertEqual(model.parameter_scale_mode, "paper")

    def test_compute_loss_adds_mapping_loss_when_enabled(self) -> None:
        model = ProjectionMappingCNN2(latent_dim=8)
        images = torch.randn(2, 1, 28, 28)
        labels = torch.tensor([0, 1])
        criterion = torch.nn.CrossEntropyLoss()
        plain_loss, plain_terms = compute_loss(
            model,
            images,
            labels,
            criterion,
            mapping_loss_weight=0.0,
        )
        mapped_loss, mapped_terms = compute_loss(
            model,
            images,
            labels,
            criterion,
            mapping_loss_weight=0.1,
            mapping_loss_coefficients=MappingLossCoefficients(1.0, 1.0, 1.0),
            mapping_perturb_std=1e-3,
        )

        self.assertEqual(plain_terms["mapping"], 0.0)
        self.assertGreater(float(mapped_terms["mapping"]), 0.0)
        self.assertGreater(float(mapped_loss.detach()), float(plain_loss.detach()))

    def test_mapping_loss_coefficients_are_trainable(self) -> None:
        coefficients = MappingLossCoefficients(0.01, 0.02, 0.03)
        values = coefficients()

        self.assertEqual(sum(parameter.numel() for parameter in coefficients.parameters()), 3)
        for value in values.values():
            self.assertTrue(value.requires_grad)
            self.assertGreater(float(value.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
