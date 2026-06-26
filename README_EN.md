# Mapping Networks Experiments

English | [中文](README.md)

Paper: [Mapping Networks](https://arxiv.org/abs/2602.19134)

This repository is a lightweight PyTorch sandbox for testing the core idea of Mapping Networks:

```text
small latent parameters z -> fixed mapping G -> generated CNN weights theta -> classification
```

In the current experiment, the latent parameters are trained from scratch. The fixed mapping is not trained, and the CNN weights are not directly optimized. Gradients pass through the generated CNN weights and the fixed mapping, then update only the latent parameters.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets are downloaded automatically by `torchvision` into `data/`.

## Main Scripts

- `experiments/train_digits.py`: trains one model on MNIST or FashionMNIST.
- `experiments/sweep_latent_dims.py`: runs multiple `latent_dim` tiers.
- `src/mapping_networks/models.py`: baseline CNNs and mapping-generated-weight models.

## Experimental Models

The main comparison uses two models.

**Original EfficientCNN**

This is a directly trained CNN baseline with 95,194 trainable parameters:

```text
input 1x28x28
-> Conv 1->24, 3x3 + GroupNorm + SiLU
-> Conv 24->48, 3x3, stride=2 + GroupNorm + SiLU
-> Conv 48->64, 3x3, stride=2 + GroupNorm + SiLU
-> Conv 64->96, 3x3 + GroupNorm + SiLU
-> AdaptiveAvgPool 1x1
-> Linear 96->10
```

This is the standard training setup: the optimizer directly updates all convolution, normalization, and linear-layer parameters.

**Projection EfficientCNN**

This model has the same target CNN architecture as EfficientCNN, so it also contains 95,194 target weights. During training, however, these CNN weights are not directly updated. Instead, a small set of latent parameters is trained, and a fixed random orthogonal projection generates the full CNN weights:

```text
latent z -> fixed projection -> generated CNN weights -> CNN forward -> loss
```

The experiment uses layer-wise latent vectors with 5 latent groups:

```text
group 1: conv1 + gn1
group 2: conv2 + gn2
group 3: conv3 + gn3
group 4: conv4 + gn4
group 5: linear classifier
```

Therefore, the trainable parameter count is:

```text
trainable_params = 5 * latent_dim
```

For example, when `latent_dim=512`, only `5 * 512 = 2,560` parameters are trained, while each forward pass still generates 95,194 CNN weights.

This is not compression of a pre-trained CNN and not distillation. The latent vector starts from random initialization and is trained directly with the MNIST classification loss until the generated CNN weights become useful.

## Reproducing Runs

Original EfficientCNN, directly training all CNN parameters:

```bash
PYTHONPATH=src python experiments/train_digits.py \
  --model efficient-cnn \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --no-progress \
  --output results/timed_full/efficient_cnn_e20.json
```

Projection EfficientCNN, training only latent parameters:

```bash
PYTHONPATH=src python experiments/train_digits.py \
  --model projection-efficient-cnn \
  --layerwise \
  --latent-dim 512 \
  --activation identity \
  --output-gain 3 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-2 \
  --no-progress \
  --output results/timed_full/mnist_ld512_e20.json
```

Run multiple latent parameter tiers:

```bash
PYTHONPATH=src python experiments/sweep_latent_dims.py \
  --latent-dims 64 128 256 512 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-2 \
  --summary results/timed_full/latent_sweep_e20_summary.md
```

`--epochs 20` means 20 training epochs, i.e. 20 full passes over the training set.

## Results

Full MNIST dataset, batch size 256, Apple MPS device. `Projection EfficientCNN` uses 5 layer-wise latent vectors, so its trainable parameter count is `5 * latent_dim`. The generated target CNN still has 95,194 weights.

| Model/tier | latent_dim | Trainable params | Target weights | Before acc | epochs | Best epoch | Best acc | Final acc | Total time | Sec/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientCNN baseline | - | 95,194 | 95,194 | 6.97% | 20 | 17 | 98.94% | 98.87% | 424.3s | 21.2s |
| Projection EfficientCNN | 64 | 320 | 95,194 | 9.28% | 20 | 20 | 73.67% | 73.67% | 505.2s | 25.3s |
| Projection EfficientCNN | 128 | 640 | 95,194 | 10.90% | 20 | 20 | 86.98% | 86.98% | 583.5s | 29.2s |
| Projection EfficientCNN | 256 | 1,280 | 95,194 | 14.07% | 20 | 19 | 92.67% | 92.58% | 627.3s | 31.4s |
| Projection EfficientCNN | 512 | 2,560 | 95,194 | 6.22% | 20 | 20 | 96.04% | 96.04% | 514.9s | 25.7s |

Observations:

- Accuracy before training is close to random guessing.
- More latent parameters improve final accuracy.
- With `latent_dim=512`, only 2,560 parameters are trained, reaching 96.04%.
- Directly training the original EfficientCNN updates 95,194 parameters and reaches a best accuracy of 98.94% after 20 epochs.

## Current Takeaway

This experiment suggests that the CNN parameter space contains substantial redundancy: instead of optimizing every CNN weight directly, it is possible to search in a lower-dimensional latent space and still generate useful CNN weights.

This is not yet a full reproduction of the paper. Main missing pieces:

- The current runs train latent parameters only with classification loss.
- Mapping Loss from the paper has not been added yet.
- FashionMNIST, larger models, and multiple random seeds have not been systematically evaluated.

So the current result should be read as a minimal validation: a low-dimensional latent vector can generate working CNN weights through a fixed mapping, but it is still short of the full paper setup.
