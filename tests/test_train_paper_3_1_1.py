import unittest

import torch

from train_paper_3_1_1 import (
    ChunkedFixedProjector,
    FixedMappingLossCoefficients,
    MappingCNN,
    TrainableMappingLossCoefficients,
    build_arg_parser,
    validate_layerwise_dims,
    validate_layerwise_modulation_scales,
    validate_paper_protocol,
)


class PaperMappingProjectorTests(unittest.TestCase):
    def test_fixed_projection_is_row_orthogonal(self) -> None:
        projector = ChunkedFixedProjector(
            latent_dim=3,
            out_dim=7,
            seed=42,
            activation="identity",
            weight_scale=1.0,
            modulation_scale=0.01,
            latent_init_std=0.02,
            projection_init="orthogonal",
            modulation_reduction="sum",
        )

        gram = projector.cached_W @ projector.cached_W.T
        torch.testing.assert_close(gram, torch.eye(3), atol=1e-5, rtol=1e-5)
        self.assertFalse(projector.cached_W.requires_grad)

    def test_legacy_blockwise_projection_initializes_each_chunk_orthogonally(self) -> None:
        projector = ChunkedFixedProjector(
            latent_dim=3,
            out_dim=8,
            seed=42,
            chunk_size=4,
            activation="identity",
            weight_scale=1.0,
            modulation_scale=0.01,
            latent_init_std=0.02,
            projection_init="orthogonal",
            modulation_reduction="mean",
            projection_layout="blockwise",
        )

        expected = torch.eye(3)
        for start in (0, 4):
            block = projector.cached_W[:, start:start + 4]
            torch.testing.assert_close(
                block @ block.T,
                expected,
                atol=1e-5,
                rtol=1e-5,
            )

        torch.testing.assert_close(
            projector.cached_W @ projector.cached_W.T,
            2.0 * expected,
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(
            float(projector.cached_W_frobenius_norm_sq),
            6.0,
        )

    def test_modulation_matches_expanded_paper_equation(self) -> None:
        projector = ChunkedFixedProjector(
            latent_dim=3,
            out_dim=5,
            seed=42,
            activation="identity",
            weight_scale=1.0,
            modulation_scale=0.2,
            latent_init_std=0.0,
            projection_init="orthogonal",
            modulation_reduction="sum",
        )
        latent = torch.tensor([1.0, 2.0, 3.0])

        with torch.no_grad():
            projector.cached_W.zero_()
            projector.cached_b.zero_()

        expected = torch.full((5,), 0.2 * torch.sum(latent * latent))
        torch.testing.assert_close(projector(latent), expected)

    def test_explicit_layerwise_dims_must_match_total(self) -> None:
        specs = [("conv1", (10,)), ("conv2", (20,)), ("fc1", (30,)), ("fc2", (40,))]

        dims = validate_layerwise_dims(specs, 12, [1, 2, 3, 6])
        self.assertEqual(dims, {"conv1": 1, "conv2": 2, "fc1": 3, "fc2": 6})

        with self.assertRaises(ValueError):
            validate_layerwise_dims(specs, 12, [3, 3, 3, 2])

    def test_layerwise_modulation_scales_follow_layer_order(self) -> None:
        scales = validate_layerwise_modulation_scales(
            ["conv1", "conv2", "fc1", "fc2"],
            default_scale=0.01,
            requested_scales=[0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(
            scales,
            {"conv1": 0.1, "conv2": 0.2, "fc1": 0.3, "fc2": 0.4},
        )

        with self.assertRaises(ValueError):
            validate_layerwise_modulation_scales(
                ["conv1", "conv2"],
                default_scale=0.01,
                requested_scales=[0.1],
            )

    def test_lwt_projectors_receive_layerwise_modulation_scales(self) -> None:
        expected = [1e-5, 2e-5, 3e-5, 4e-5]
        model = MappingCNN(
            model_name="cnn2",
            mode="lwt",
            latent_dim=None,
            total_latent_dim=4,
            seed=42,
            chunk_size=4096,
            activation="tanh",
            weight_scale=1.0,
            modulation_scale=0.01,
            latent_init_std=0.02,
            projection_init="orthogonal",
            modulation_reduction="sum",
            parameter_scale_mode="paper",
            layerwise_latent_dims=[1, 1, 1, 1],
            layerwise_modulation_scales=expected,
        )

        actual = [
            projector.modulation_scale
            for projector in model.projectors.values()
        ]
        self.assertEqual(actual, expected)

    def test_smoothness_matches_autograd_jacobian(self) -> None:
        projector = ChunkedFixedProjector(
            latent_dim=3,
            out_dim=5,
            seed=7,
            activation="tanh",
            weight_scale=1.3,
            modulation_scale=0.04,
            latent_init_std=0.2,
            projection_init="orthogonal",
            modulation_reduction="sum",
        )

        jacobian = torch.autograd.functional.jacobian(
            lambda latent: projector(latent),
            projector.z,
            create_graph=True,
        )
        expected = torch.sum(jacobian * jacobian)
        torch.testing.assert_close(
            projector.smoothness_loss(),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_alignment_uses_modulated_mapping_weights(self) -> None:
        projector = ChunkedFixedProjector(
            latent_dim=3,
            out_dim=5,
            seed=11,
            activation="identity",
            weight_scale=1.0,
            modulation_scale=0.2,
            latent_init_std=0.0,
            projection_init="orthogonal",
            modulation_reduction="sum",
        )
        with torch.no_grad():
            projector.z.copy_(torch.tensor([1.0, 2.0, 3.0]))
            projector.cached_W.zero_()
            projector.cached_W_row_mean.zero_()

        torch.testing.assert_close(projector.alignment_loss(), torch.tensor(0.0))

    def test_alignment_uses_same_reduction_as_forward_mapping(self) -> None:
        projector = ChunkedFixedProjector(
            latent_dim=2,
            out_dim=3,
            seed=13,
            activation="identity",
            weight_scale=1.0,
            modulation_scale=0.2,
            latent_init_std=0.0,
            projection_init="orthogonal",
            modulation_reduction="mean",
        )
        with torch.no_grad():
            projector.z.copy_(torch.tensor([1.0, 2.0]))
            projector.cached_W_row_mean.copy_(torch.tensor([0.4, -0.1]))

        expected_row_mean = (
            projector.cached_W_row_mean
            + (projector.modulation_scale / projector.latent_dim) * projector.z
        )
        expected = 1.0 - torch.nn.functional.cosine_similarity(
            projector.z.unsqueeze(0),
            expected_row_mean.unsqueeze(0),
            dim=1,
        ).mean()
        torch.testing.assert_close(projector.alignment_loss(), expected)

    def test_mapping_loss_coefficients_are_positive_and_trainable(self) -> None:
        coefficients = TrainableMappingLossCoefficients(0.05, 1e-4, 0.01)
        values = coefficients()
        loss = sum(values.values())
        loss.backward()

        for parameter in coefficients.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad), 0.0)
        self.assertAlmostEqual(float(values["stability"].detach()), 0.05, places=6)
        self.assertAlmostEqual(float(values["smoothness"].detach()), 1e-4, places=8)
        self.assertAlmostEqual(float(values["alignment"].detach()), 0.01, places=6)

    def test_fixed_mapping_loss_coefficients_disable_unselected_terms(self) -> None:
        coefficients = FixedMappingLossCoefficients(
            0.05,
            1e-4,
            0.01,
            enabled_terms=["smoothness", "alignment"],
        )

        self.assertEqual(sum(p.numel() for p in coefficients.parameters()), 0)
        values = coefficients.detached_values()
        self.assertEqual(values["stability"], 0.0)
        self.assertAlmostEqual(values["smoothness"], 1e-4, places=8)
        self.assertAlmostEqual(values["alignment"], 0.01, places=6)

    def test_trainable_mapping_loss_only_optimizes_selected_terms(self) -> None:
        coefficients = TrainableMappingLossCoefficients(
            0.05,
            1e-4,
            0.01,
            enabled_terms=["stability"],
        )

        self.assertEqual(sum(p.numel() for p in coefficients.parameters()), 1)
        values = coefficients()
        self.assertGreater(float(values["stability"].detach()), 0.0)
        self.assertEqual(float(values["smoothness"].detach()), 0.0)
        self.assertEqual(float(values["alignment"].detach()), 0.0)

    def test_parser_accepts_fixed_mapping_loss_ablation(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--loss-mode",
                "full",
                "--loss-coefficient-mode",
                "fixed",
                "--mapping-loss-terms",
                "smoothness",
                "alignment",
            ]
        )

        self.assertEqual(args.loss_coefficient_mode, "fixed")
        self.assertEqual(args.mapping_loss_terms, ["smoothness", "alignment"])

    def test_parser_accepts_layerwise_modulation_scales(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--layerwise-modulation-scales",
                "1e-5",
                "2e-5",
                "3e-5",
                "4e-5",
            ]
        )

        self.assertEqual(
            args.layerwise_modulation_scales,
            [1e-5, 2e-5, 3e-5, 4e-5],
        )

    def test_paper_protocol_accepts_disclosed_full_configuration(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--loss-mode",
                "full",
                "--projection-init",
                "orthogonal",
                "--modulation-reduction",
                "sum",
                "--parameter-scale-mode",
                "paper",
                "--weight-scale",
                "1",
                "--loss-coefficient-mode",
                "trainable",
            ]
        )

        validate_paper_protocol(args)

    def test_paper_protocol_rejects_mean_modulation(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--modulation-reduction",
                "mean",
            ]
        )

        with self.assertRaisesRegex(ValueError, "modulation-reduction sum"):
            validate_paper_protocol(args)

    def test_paper_protocol_rejects_legacy_blockwise_projection(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--projection-layout",
                "blockwise",
            ]
        )

        with self.assertRaisesRegex(ValueError, "projection-layout global"):
            validate_paper_protocol(args)

    def test_paper_protocol_rejects_fixed_full_loss_coefficients(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--loss-mode",
                "full",
                "--loss-coefficient-mode",
                "fixed",
            ]
        )

        with self.assertRaisesRegex(ValueError, "trainable"):
            validate_paper_protocol(args)

    def test_paper_protocol_rejects_unreported_table1_layerwise_alphas(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--layerwise-modulation-scales",
                "1e-5",
                "2e-5",
                "3e-5",
                "4e-5",
            ]
        )

        with self.assertRaisesRegex(ValueError, "does not report them for Table 1"):
            validate_paper_protocol(args)

    def test_custom_protocol_allows_controlled_ablation(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "mapping",
                "--protocol",
                "custom",
                "--loss-mode",
                "full",
                "--modulation-reduction",
                "mean",
                "--loss-coefficient-mode",
                "fixed",
                "--mapping-loss-terms",
                "smoothness",
            ]
        )

        validate_paper_protocol(args)


if __name__ == "__main__":
    unittest.main()
