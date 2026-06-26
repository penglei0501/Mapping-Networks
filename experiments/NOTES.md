# Experiment notes

## Paper files

- PDF: `papers/2602.19134.pdf`
- arXiv source: `papers/source/`

## What the paper's method says

The paper's Mapping Network trains a low-dimensional latent vector `z` instead of the
target network parameters. A fixed, orthogonally initialized mapping generates the full
target parameter vector. The paper also uses weight modulation:

```text
W_ij <- W_ij + alpha * z_i
theta_hat = activation(W z + b)
```

The paper describes two training strategies:

- SLVT: one latent vector generates all target-network parameters.
- LWT: one latent vector per layer, which is more memory-friendly and usually easier to optimize.

## Current sandbox

Implemented:

- `DirectMLP`: normal directly trained target network baseline.
- `DirectCNN`: small CNN baseline with 105,866 trainable parameters.
- `EfficientCNN`: stronger compact CNN with 95,194 trainable parameters. It reaches
  97.42% on full MNIST after 3 epochs.
- `MappingMLP`: coordinate-to-scalar mapper, useful as a debugging contrast.
- `ProjectionMappingMLP`: fixed projection mapper closer to the paper's description.
- `ProjectionMappingCNN`: fixed projection mapper that generates a small CNN target.
- `ProjectionMappingEfficientCNN`: mapped version of the stronger compact CNN.

The first two low-parameter projection runs show real learning but not paper-level
accuracy yet. That is expected for this first pass because we are mapping an MLP target,
while the paper reports MNIST/FashionMNIST results on CNN targets with mapping loss.

The first CNN target result is:

- Direct CNN quick, 2 epochs: 62.01% with 105,866 trainable parameters.
- Projection CNN quick, 5 epochs: 36.72% with 2,048 trainable latent parameters.
- Projection CNN init-only: 11.52%, so the mapped CNN is learning from latent training.
- Efficient CNN full MNIST, 3 epochs: 97.42% with 95,194 trainable parameters.
- Projection Efficient CNN quick, 5 epochs: 35.35% with 2,560 trainable latent parameters.
- Projection Efficient CNN full MNIST, 20 epochs: 96.04% with 2,560 trainable latent
  parameters and 95,194 generated target parameters.
- Projection Efficient CNN init-only accuracy on full MNIST test set is near random;
  the exact value depends on `latent_dim` because each tier initializes a different
  generated target network.

## Parameter tiers

Because `ProjectionMappingEfficientCNN` uses five layer-wise latent vectors, the
trainable parameter count is `5 * latent_dim`. These runs use full MNIST, batch size
256, AdamW, `lr=1e-2`, `activation=identity`, and `output_gain=3`.

| model/tier | latent_dim | trainable params | target params | before acc | epochs | best epoch | best acc | final acc | total time | sec/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Efficient CNN baseline | - | 95,194 | 95,194 | 6.97% | 20 | 17 | 98.94% | 98.87% | 424.3s | 21.2s |
| Projection Efficient CNN | 64 | 320 | 95,194 | 9.28% | 20 | 20 | 73.67% | 73.67% | 505.2s | 25.3s |
| Projection Efficient CNN | 128 | 640 | 95,194 | 10.90% | 20 | 20 | 86.98% | 86.98% | 583.5s | 29.2s |
| Projection Efficient CNN | 256 | 1,280 | 95,194 | 14.07% | 20 | 19 | 92.67% | 92.58% | 627.3s | 31.4s |
| Projection Efficient CNN | 512 | 2,560 | 95,194 | 6.22% | 20 | 20 | 96.04% | 96.04% | 514.9s | 25.7s |

The 512-tier run was repeated with `--num-workers 0` after a PyTorch DataLoader shared
memory timeout with worker processes. The model result matched the previous untimed
20-epoch run at 96.04%.

## Recommended next runs

Run quick MNIST first:

```bash
PYTHONPATH=src python experiments/train_digits.py --model projection-cnn --layerwise --latent-dim 512 --activation identity --output-gain 3 --quick --epochs 20 --lr 1e-2 --output results/projection_cnn_layerwise_gain3_e20.json
```

Run the stronger mapped CNN on full MNIST:

```bash
PYTHONPATH=src python experiments/train_digits.py --model projection-efficient-cnn --layerwise --latent-dim 512 --activation identity --output-gain 3 --epochs 20 --batch-size 256 --lr 1e-2 --output results/projection_efficient_cnn_full_e20.json
```

Then compare FashionMNIST:

```bash
PYTHONPATH=src python experiments/train_digits.py --dataset fashion --model projection --layerwise --latent-dim 512 --activation identity --output-gain 10 --quick --epochs 20 --lr 1e-2 --output results/projection_layerwise_fashion_e20.json
```

For fuller runs, remove `--quick` and increase epochs.
