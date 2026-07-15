import unittest

import torch

from train_paper_3_1_1 import count_trainable_params
from train_paper_3_6_table8 import (
    TABLE_COUNT_MATCHING_RANK,
    TABLE_LRD_PARAMS,
    TABLE_PRUNED_PARAMS,
    TARGET_DENSE_PARAMS,
    LowRankCNN2,
    apply_global_pruning,
    apply_tensor_masks,
    count_nonzero_tensors,
    count_specs,
    dense_specs,
    exact_global_magnitude_masks,
    exact_tensorwise_magnitude_masks,
    enforce_parameter_masks,
    functional_addon_forward,
    low_rank_specs,
    masked_params,
    register_parameter_mask_hooks,
)


class Table8Tests(unittest.TestCase):
    def test_dense_parameter_count_matches_cnn2(self) -> None:
        self.assertEqual(count_specs(dense_specs()), TARGET_DENSE_PARAMS)

    def test_rank_32_exactly_matches_table_parameter_count(self) -> None:
        self.assertEqual(TABLE_COUNT_MATCHING_RANK, 32)
        self.assertEqual(count_specs(low_rank_specs(32)), TABLE_LRD_PARAMS)
        self.assertEqual(count_trainable_params(LowRankCNN2(32)), TABLE_LRD_PARAMS)

    def test_rank_16_does_not_match_table_parameter_count(self) -> None:
        self.assertEqual(count_specs(low_rank_specs(16)), 21_066)
        self.assertNotEqual(count_specs(low_rank_specs(16)), TABLE_LRD_PARAMS)

    def test_global_masks_keep_exact_count(self) -> None:
        tensors = {
            "a": torch.arange(12, dtype=torch.float32),
            "b": torch.arange(12, 30, dtype=torch.float32),
        }
        masks = exact_global_magnitude_masks(tensors, keep_count=7)
        self.assertEqual(sum(int(mask.sum()) for mask in masks.values()), 7)

    def test_table_pruning_keeps_exactly_10862_values(self) -> None:
        tensors = {
            "a": torch.arange(TARGET_DENSE_PARAMS, dtype=torch.float32)
        }
        masked, _ = masked_params(tensors, TABLE_PRUNED_PARAMS)
        self.assertEqual(count_nonzero_tensors(masked), TABLE_PRUNED_PARAMS)

    def test_tensorwise_cnn2_pruning_keeps_exact_table_count(self) -> None:
        tensors = {
            name: torch.randn(shape)
            for name, shape in dense_specs()
        }
        masks = exact_tensorwise_magnitude_masks(tensors, keep_fraction=0.1)
        self.assertEqual(
            sum(int(mask.sum()) for mask in masks.values()),
            TABLE_PRUNED_PARAMS,
        )
        self.assertTrue(all(int(mask.sum()) > 0 for mask in masks.values()))

    def test_fixed_pruning_masks_survive_optimizer_step(self) -> None:
        torch.manual_seed(7)
        model = torch.nn.Linear(4, 2)
        masks = apply_global_pruning(model, keep_count=3)
        handles = register_parameter_mask_hooks(model, masks)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.randn(5, 4)).square().mean()
        loss.backward()
        optimizer.step()
        enforce_parameter_masks(model, masks)
        for handle in handles:
            handle.remove()
        self.assertEqual(
            count_nonzero_tensors(dict(model.named_parameters())),
            3,
        )

    def test_generated_parameter_mask_blocks_pruned_gradients(self) -> None:
        value = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        masked = apply_tensor_masks(
            {"value": value},
            {"value": torch.tensor([True, False, True])},
        )
        masked["value"].sum().backward()
        torch.testing.assert_close(
            value.grad,
            torch.tensor([1.0, 0.0, 1.0]),
        )

    def test_low_rank_functional_forward_matches_module(self) -> None:
        torch.manual_seed(4)
        model = LowRankCNN2(rank=32).eval()
        params = {
            "conv1.weight": model.conv1.weight,
            "conv1.bias": model.conv1.bias,
            "conv2.weight": model.conv2.weight,
            "conv2.bias": model.conv2.bias,
            "fc1.v": model.fc1_v.weight,
            "fc1.u": model.fc1_u.weight,
            "fc1.bias": model.fc1_u.bias,
            "fc2.weight": model.fc2.weight,
            "fc2.bias": model.fc2.bias,
        }
        x = torch.randn(3, 1, 28, 28)
        torch.testing.assert_close(
            functional_addon_forward(x, params, low_rank=True),
            model(x),
        )


if __name__ == "__main__":
    unittest.main()
