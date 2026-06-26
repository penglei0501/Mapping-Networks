# Mapping Networks 复现实验

论文地址：[Mapping Networks](https://arxiv.org/abs/2602.19134)

这个仓库是一个轻量 PyTorch 实验，用来验证 Mapping Networks 的核心想法：

```text
少量 latent 参数 z -> 固定映射 G -> 生成完整 CNN 参数 theta -> 用 theta 做分类
```

当前实验不是先训练好 CNN 再压缩它，而是从头训练 latent 参数。固定映射不训练，CNN 权重也不直接训练；反向传播会经过生成出来的 CNN 权重和固定映射，最后只更新 latent 参数。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

数据集会由 `torchvision` 自动下载到 `data/`。

## 主要脚本

- `experiments/train_digits.py`：训练单个模型，支持 MNIST/FashionMNIST。
- `experiments/sweep_latent_dims.py`：按多个 `latent_dim` 批量跑参数档位。
- `src/mapping_networks/models.py`：普通 CNN、EfficientCNN、Mapping 生成权重版本。

## 实验模型

本次主要比较两个模型。

**EfficientCNN 原始模型**

这是一个直接训练 CNN 权重的基线模型，共 95,194 个可训练参数：

```text
输入 1x28x28
-> Conv 1->24, 3x3 + GroupNorm + SiLU
-> Conv 24->48, 3x3, stride=2 + GroupNorm + SiLU
-> Conv 48->64, 3x3, stride=2 + GroupNorm + SiLU
-> Conv 64->96, 3x3 + GroupNorm + SiLU
-> AdaptiveAvgPool 1x1
-> Linear 96->10
```

它是普通训练方式：优化器直接更新全部卷积、归一化和线性层参数。

**Projection EfficientCNN**

这个模型的目标 CNN 结构和 EfficientCNN 完全相同，目标权重数也是 95,194。但训练时不直接更新这些 CNN 权重，而是训练少量 latent 参数，再通过固定随机正交投影生成完整 CNN 权重：

```text
latent z -> fixed projection -> generated CNN weights -> CNN forward -> loss
```

实验使用 layer-wise latent，共 5 组 latent：

```text
第 1 组：conv1 + gn1
第 2 组：conv2 + gn2
第 3 组：conv3 + gn3
第 4 组：conv4 + gn4
第 5 组：linear classifier
```

所以可训练参数量是：

```text
trainable_params = 5 * latent_dim
```

例如 `latent_dim=512` 时，只训练 `5 * 512 = 2,560` 个参数，但每次 forward 都会生成 95,194 个 CNN 权重。

这不是把训练好的 CNN 参数拿来压缩，也不是蒸馏；它是从随机初始化 latent 开始，用 MNIST 分类 loss 直接训练 latent，让生成出来的 CNN 权重逐渐变得可用。

## 复现实验

原始 EfficientCNN，直接训练全部 CNN 参数：

```bash
PYTHONPATH=src python experiments/train_digits.py \
  --model efficient-cnn \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --no-progress \
  --output results/timed_full/efficient_cnn_e20.json
```

Projection EfficientCNN，只训练 latent 参数：

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

跑多个 latent 参数档位：

```bash
PYTHONPATH=src python experiments/sweep_latent_dims.py \
  --latent-dims 64 128 256 512 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-2 \
  --summary results/timed_full/latent_sweep_e20_summary.md
```

`--epochs 20` 的意思是训练 20 轮，也就是完整遍历训练集 20 次。

## 实验结果

MNIST full dataset，batch size 256，Apple MPS 设备。`Projection EfficientCNN` 使用 5 个 layer-wise latent 向量，所以可训练参数量是 `5 * latent_dim`。生成出来的目标 CNN 仍然有 95,194 个权重。

| 模型/档位 | latent_dim | 可训练参数 | 目标权重数 | 训练前 acc | epochs | 最佳轮 | 最佳 acc | 最终 acc | 总时长 | 每轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientCNN 原始训练 | - | 95,194 | 95,194 | 6.97% | 20 | 17 | 98.94% | 98.87% | 424.3s | 21.2s |
| Projection EfficientCNN | 64 | 320 | 95,194 | 9.28% | 20 | 20 | 73.67% | 73.67% | 505.2s | 25.3s |
| Projection EfficientCNN | 128 | 640 | 95,194 | 10.90% | 20 | 20 | 86.98% | 86.98% | 583.5s | 29.2s |
| Projection EfficientCNN | 256 | 1,280 | 95,194 | 14.07% | 20 | 19 | 92.67% | 92.58% | 627.3s | 31.4s |
| Projection EfficientCNN | 512 | 2,560 | 95,194 | 6.22% | 20 | 20 | 96.04% | 96.04% | 514.9s | 25.7s |

可以看到：

- 训练前准确率基本接近随机猜测。
- latent 参数越多，最终准确率越高。
- `latent_dim=512` 时，只训练 2,560 个参数，可以达到 96.04%。
- 原始 EfficientCNN 直接训练 95,194 个参数，20 epochs 后达到 98.94% 最佳准确率。

## 当前结论

这个实验说明 CNN 参数空间里存在明显冗余：不一定要直接优化所有 CNN 权重，也可以在较低维 latent 空间中寻找一组可用的生成权重。

但当前实现还不是论文的完整复现，主要差别是：

- 目前只用分类 loss 训练 latent。
- 还没有加入论文里的 Mapping Loss。
- 还没有系统比较 FashionMNIST、更大模型和多随机种子结果。

因此这里的结果更适合作为一个最小验证：低维 latent 确实能通过固定映射生成可工作的 CNN 参数，但和完整论文结果还有距离。
