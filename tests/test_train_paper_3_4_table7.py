import unittest

import torch

from train_paper_3_1_1 import count_trainable_params
from train_paper_3_4_table7 import (
    AssumedFullDNNProjector,
    SeparateModulationProjector,
    TARGET_PARAM_COUNT,
    TrainableLinearProjector,
    build_arg_parser,
    resolve_variant_spec,
    validate_protocol,
)


class Table7RobustnessTests(unittest.TestCase):
    def test_variant_specs_resolve_reported_parameter_budgets(self) -> None:
        self.assertEqual(
            resolve_variant_spec("ours", 1024, "mnist").latent_dim,
            1024,
        )
        lv_wmap = resolve_variant_spec("lv-wmap", 4096, "fmnist")
        self.assertEqual(lv_wmap.latent_dim, 2048)
        self.assertEqual(lv_wmap.modulation_dim, 2048)
        self.assertEqual(
            resolve_variant_spec("lv-full-dnn", 543095, "mnist").latent_dim,
            5,
        )

    def test_unreported_parameter_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_variant_spec("ours", 512, "mnist")
        with self.assertRaises(ValueError):
            resolve_variant_spec("lv-wmap", None, "mnist")

    def test_separate_modulation_uses_independent_vector(self) -> None:
        projector = SeparateModulationProjector(
            latent_dim=3,
            out_dim=5,
            seed=7,
            chunk_size=8,
            activation_name="identity",
            weight_scale=1.0,
            modulation_scale=0.2,
            latent_init_std=0.0,
            projection_init="orthogonal",
            modulation_reduction="sum",
            projection_layout="global",
        )
        with torch.no_grad():
            projector.core.cached_W.zero_()
            projector.core.cached_b.zero_()
            projector.core.z.copy_(torch.tensor([1.0, 2.0, 3.0]))
            projector.modulation.copy_(torch.tensor([4.0, 5.0, 6.0]))

        expected = torch.full((5,), 0.2 * (4.0 + 10.0 + 18.0))
        torch.testing.assert_close(projector(), expected)
        self.assertEqual(count_trainable_params(projector), 6)

    def test_separate_modulation_smoothness_matches_jacobian(self) -> None:
        projector = SeparateModulationProjector(
            latent_dim=3,
            out_dim=5,
            seed=11,
            chunk_size=8,
            activation_name="tanh",
            weight_scale=1.2,
            modulation_scale=0.03,
            latent_init_std=0.1,
            projection_init="orthogonal",
            modulation_reduction="sum",
            projection_layout="global",
        )
        jacobian = torch.autograd.functional.jacobian(
            lambda latent: projector(latent),
            projector.latent_vector,
            create_graph=True,
        )
        expected = torch.sum(jacobian.square())
        torch.testing.assert_close(
            projector.smoothness_loss(),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_lv_full_dnn_parameter_counts_are_exact(self) -> None:
        for latent_dim, expected in ((5, 543095), (15, 1629285)):
            projector = TrainableLinearProjector(
                latent_dim=latent_dim,
                out_dim=TARGET_PARAM_COUNT,
                seed=42,
                activation_name="identity",
                weight_scale=1.0,
                latent_init_std=0.02,
                projection_init="orthogonal",
            )
            self.assertEqual(count_trainable_params(projector), expected)
            self.assertEqual(tuple(projector(torch.zeros(latent_dim)).shape), (108618,))

    def test_lv_full_dnn_gaussian_projection_has_unit_column_scale(self) -> None:
        projector = TrainableLinearProjector(
            latent_dim=5,
            out_dim=4096,
            seed=42,
            activation_name="identity",
            weight_scale=1.0,
            latent_init_std=0.02,
            projection_init="gaussian",
        )
        column_norms = torch.linalg.vector_norm(projector.mapping_weight, dim=0)
        self.assertGreater(column_norms.mean().item(), 0.8)
        self.assertLess(column_norms.mean().item(), 1.2)

    def test_full_dnn_assumption_matches_only_the_reported_count(self) -> None:
        projector = AssumedFullDNNProjector(
            out_dim=TARGET_PARAM_COUNT,
            seed=42,
            activation_name="identity",
            latent_init_std=0.02,
        )
        self.assertEqual(count_trainable_params(projector), 6753104)
        self.assertFalse(projector.fixed_z.requires_grad)

    def test_paper_protocol_rejects_undisclosed_full_dnn(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--variant", "full-dnn"])
        spec = resolve_variant_spec("full-dnn", None, "mnist")
        with self.assertRaisesRegex(ValueError, "does not disclose"):
            validate_protocol(args, spec)

    def test_custom_full_dnn_requires_explicit_acknowledgement(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--variant", "full-dnn", "--protocol", "custom"]
        )
        spec = resolve_variant_spec("full-dnn", None, "mnist")
        with self.assertRaisesRegex(ValueError, "allow-undisclosed"):
            validate_protocol(args, spec)

        args.allow_undisclosed_full_dnn = True
        validate_protocol(args, spec)

    def test_baseline_rejects_mapping_loss(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--variant", "baseline", "--loss-mode", "full"]
        )
        spec = resolve_variant_spec("baseline", None, "mnist")
        with self.assertRaisesRegex(ValueError, "cross-entropy"):
            validate_protocol(args, spec)


if __name__ == "__main__":
    unittest.main()
