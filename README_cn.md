简体中文 | [English](README.md)

**Salience DETR**: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement
===

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/salience-detr-enhancing-detection-transformer-1/object-detection-on-coco-2017-val)](https://paperswithcode.com/sota/object-detection-on-coco-2017-val?p=salience-detr-enhancing-detection-transformer-1)
[![arXiv](https://img.shields.io/badge/arXiv-2403.16131-b31b1b.svg)](https://arxiv.org/abs/2403.16131)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![GitHub license](https://img.shields.io/github/license/xiuqhou/Salience-DETR.svg?color=blue)](https://github.com/xiuqhou/Salience-DETR/blob/master/LICENSE)
![GitHub stars](https://img.shields.io/github/stars/xiuqhou/Salience-DETR)
![GitHub forks](https://img.shields.io/github/forks/xiuqhou/Salience-DETR)

本仓库是**CVPR 2024**（得分**553**）论文[Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html)的官方实现. 作者：[Xiuquan Hou](https://github.com/xiuqhou), [Meiqin Liu](https://scholar.google.com/citations?user=T07OWMkAAAAJ&hl=zh-CN&oi=ao), Senlin Zhang, [Ping Wei](https://scholar.google.com/citations?user=1OQBtdcAAAAJ&hl=zh-CN&oi=ao), [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).

💖 如果我们的Salience-DETR有帮到您的研究或项目，请为本仓库点颗star，谢谢！🤗

<div align="center">
    <img src="images/Salience-DETR.svg">
</div>

<details>

<summary>✨研究亮点</summary>

1. 我们深入分析了两阶段DETR类方法中存在的[尺度偏差和查询冗余](id_1)问题。
2. 我们提出了一种在显著性监督下降低计算复杂度的分层过滤机制，所提出的监督方式甚至能在仅使用检测框标注的情况下捕捉[细粒度的物体轮廓](#id_2)。
3. Salience DETR在三个极具挑战的缺陷检测任务上分别提升了 **+4.0%**, **+0.2%** 和 **+4.4%** AP，在COCO 2017上只使用了大约 **70\%** FLOPs 实现了相当的精度。

</details>

<details>

<summary>🔎可视化</summary>

- 现有DETR方法的两阶段选择出的查询通常是**冗余**的，并且存在**尺度偏执**（左图）。
- 对于缺陷检测和目标检测任务，**显著性监督**都有助于在仅使用检测框标注的情况下捕捉**物体轮廓**（右图）.

<h3 align="center">
    <a id="id_1"><img src="images/query_visualization.svg" width="335"></a>
    <a id="id_2"><img src="images/salience_visualization.svg" width="462"></a>
</h3>

</details>

## 更新动态

- [2024-07-18] 我们发布了[Relation-DETR](https://github.com/xiuqhou/Relation-DETR)，一个通用且强大的目标检测模型，只需要**2个epoch即可在COCO上达到40+%的AP**，性能超越[DDQ-DETR](https://github.com/jshilong/DDQ/tree/ddq_detr), [StableDINO](https://github.com/idea-research/stable-dino), [Rank-DETR](https://github.com/LeapLabTHU/Rank-DETR), [MS-DETR](https://github.com/Atten4Vis/MS-DETR)等大多数SOTA方法。代码和权重[在此](https://github.com/xiuqhou/Relation-DETR)。

- [2024-04-19] 以 [FocalNet-Large](https://github.com/microsoft/FocalNet) 作为主干网，Salience DETR在COCO val2017上取得了 **56.8 AP**, [**配置**](configs/salience_detr/salience_detr_focalnet_large_lrf_800_1333.py) 和 [**权重**](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_focalnet_large_lrf_800_1333_coco_1x.pth) 已更新!

- [2024-04-08] 更新以ConvNeXt-L作为主干网、在COCO 2017数据集上训练12轮的Salience DETR [**配置**](configs/salience_detr/salience_detr_convnext_l_800_1333.py) 和 [**权重**](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_convnext_l_800_1333_coco_1x.pth).

- [2024-04-01] 使用Swin-L作为主干网，Salience DETR在COCO 2017数据集上取得了 **56.5** AP (训练12轮)。 模型 [**配置**](configs/salience_detr/salience_detr_swin_l_800_1333.py) 和 [**权重**](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_swin_l_800_1333_coco_1x.pth) 已发布.

- [2024-03-26] 我们发布了Salience DETR的代码和在COCO 2017上使用ResNet50作为主干网络的预训练权重。

- [2024-02-29] Salience DETR被CVPR2024接受，欢迎关注！

## 模型库

在被 **CVPR 2024** 接受以后, 我们又在多种设置下重新训练了以 **ResNet50** 和 **Swin-L** 作为主干网的 **Salience DETR** 。我们提供了相应的 [**COCO 2017**](https://cocodataset.org/#home) 数据集的配置和权重。

### 训练12轮

| 模型          | 主干网                  |  AP   | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |                                                                                                     下载                                                                                                     |
| ------------- | ----------------------- | :---: | :-------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Salience DETR | ResNet50                | 50.0  |   67.7    |   54.2    |   33.3   |   54.4   |   64.4   |           [配置](configs/salience_detr/salience_detr_resnet50_800_1333.py) / [权重](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_resnet50_800_1333_coco_1x.pth)           |
| Salience DETR | ConvNeXt-L              | 54.2  |   72.4    |   59.1    |   38.8   |   58.3   |   69.6   |         [配置](configs/salience_detr/salience_detr_convnext_l_800_1333.py) / [权重](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_convnext_l_800_1333_coco_1x.pth)         |
| Salience DETR | Swin-L<sub>(IN-22K)     | 56.5  |   75.0    |   61.5    |   40.2   |   61.2   |   72.8   |             [配置](configs/salience_detr/salience_detr_swin_l_800_1333.py) / [权重](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_swin_l_800_1333_coco_1x.pth)             |
| Salience DETR | FocalNet-L<sub>(IN-22K) | 57.3  |   75.5    |   62.3    |   40.9   |   61.8   |   74.5   | [配置](configs/salience_detr/salience_detr_focalnet_large_lrf_800_1333.py) / [权重](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_focalnet_large_lrf_800_1333_coco_1x.pth) |

### 训练24轮

| 模型          | 主干网                  |  AP   | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |                                                                                                     下载                                                                                                     |
| ------------- | ----------------------- | :---: | :-------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Salience DETR | ResNet50                | 51.2  |   68.9    |   55.7    |   33.9   |   55.5   |   65.6   |           [配置](configs/salience_detr/salience_detr_resnet50_800_1333.py) / [权重](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_resnet50_800_1333_coco_2x.pth)           |

## 🔧安装步骤

1. 克隆本仓库：

    ```shell
    git clone https://github.com/xiuqhou/Salience-DETR.git
    cd Salience-DETR/
    ```

2. 创建并激活conda环境：

    ```shell
    conda create -n salience_detr python=3.8
    conda activate salience_detr
    ```

3. 根据官方步骤 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 安装pytorch。本代码要求 `python>=3.8, torch>=1.11.0, torchvision>=0.12.0`。

    ```shell
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    ```

4. 安装其他依赖：

    ```shell
    conda install --file requirements.txt -c conda-forge
    ```

您不需要手动编译CUDA算子，代码第一次运行时会自动编译并加载。

## 📁准备数据集

请按照如下格式下载 [COCO 2017](https://cocodataset.org/) 数据集或准备您自己的数据集，并将他们放在 `data/` 目录下。您可以使用 [`tools/visualize_datasets.py`](tools/visualize_datasets.py) 来可视化数据集以验证其正确性。

```shell
coco/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

<details>

<summary>可视化例子</summary>

```shell
python tools/visualize_datasets.py \
    --coco-img data/coco/val2017 \
    --coco-ann data/coco/annotations/instances_val2017.json \
    --show-dir visualize_dataset/
```

</details>

## 📚︎训练模型

我们使用 `accelerate` 包来原生处理多GPU训练，您只需要使用 `CUDA_VISIBLE_DEVICES` 来指定要用于训练的GPU/GPUs。如果未指定，脚本会自动使用机器上所有可用的GPU来训练。

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py    # 使用1个GPU进行训练
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py  # 使用2个GPU进行训练
```

训练之前请调整 [`configs/train_config.py`](configs/train_config.py) 中的参数。

<details>

<summary>训练配置文件的例子</summary>

```python
from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# 经常需要改动的训练配置
num_epochs = 12   # 训练轮次
batch_size = 2    # 总批次尺寸 = GPU数量 x 批次尺寸batch_size
num_workers = 4   # pytorch DataLoader加载数据所使用的进程数量
pin_memory = True # 是否在 pytorch DataLoader 中使用pin_memory
print_freq = 50   # 日志记录的频率
starting_epoch = 0
max_norm = 0.1    # 梯度裁剪的范数

output_dir = None  # 保存checkpoints的路径，如果设置为None，则默认保存至checkpoints/{model_name}路径下
find_unused_parameters = False  # 用于调试分布式训练

# 定义用于训练的数据集
coco_path = "data/coco"  # 数据集路径
train_transform = presets.detr  # 从 transforms/presets.py 文件中选择数据增强
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2017.json",
    transforms=None,  # eval_transform已集成至网络前向传播中
)

# 模型配置文件
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"

# 指定一个检查点文件夹来恢复训练，或者指定一个“.pth”文件来进行微调，例如：
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = None

learning_rate = 1e-4  # 初始学习率
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# 为不同的参数定义不同学习率
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
```
</details>

## 📈评估和测试

为了使用单个或多个GPU来评估模型，请指定 `CUDA_VISIBLE_DEVICES`、`dataset`、`model`、`checkpoint` 等参数。

```shell
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch test.py --coco-path /path/to/coco --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth
```

以下是可选参数，更多参数请查看 [test.py](test.py) 。

- `--show-dir`: 指定用于保存可视化结果的文件夹路径。
- `--result`: 指定用于保存检测结果的文件路径，必须以 `.json` 结尾。

<details>

<summary>模型评估的例子</summary>

例如，使用8张GPU来在 `coco` 上评估 `salience_detr_resnet50_800_1333` 模型，并将检测结果保存至 `result.json` 文件，并将检测结果的可视化保存至 `visualization/` 文件夹下，请运行以下命令：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch test.py
    --coco-path data/coco \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-21_29_56/best_ap.pth \
    --result result.json \
    --show-dir visualization/
```

</details>

<details>

<summary>评估json结果文件</summary>

在获取到上述保存的json检测结果文件后，如果要对该文件进行评估，请指定 `--result` 参数但不需要指定 `--model` 参数。

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py --coco-path /path/to/coco --result /path/to/result.json
```

以下是可选参数，完整参数请查看 [test.py](test.py) ：

- `--show-dir`: 指定用于保存可视化结果的文件夹路径。

</details>

## ▶︎模型推理

使用 [`inference.py`](inference.py) 来推理图片，使用 `--image-dir` 指定图片所在的文件夹路径。

```shell
python inference.py --image-dir /path/to/images --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth --show-dir /path/to/dir
```

<details>

<summary>推理图片的例子</summary>

例如，运行如下命令推理 `images/` 文件夹下的图片并将可视化结果保存至 `visualization/` 文件夹中。

```shell
python inference.py \
    --image-dir images/ \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint checkpoint.pth \
    --show-dir visualization/
```

</details>

或使用 [`inference.ipynb`](inference.ipynb) 进行单张图片的推理和可视化。

## 🔁评估模型速度、显存和参数

使用 `tools/benchmark_model.py` 来评估模型的推理速度、显存占用和参数量。

```shell
python tools/benchmark_model.py --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py
```

## 📍训练自定义的数据集

训练您自己的数据集之前请执行下面步骤：

1. 按照COCO标注格式准备您自己的数据集，并相应地将 [`configs/train_config.py`](configs/train_config.py) 中的 `coco_path` 参数调整为数据集所在的路径。
2. 打开 [`configs/salience_detr`](configs/salience_detr) 路径下的模型配置文件，将 `num_classes` 参数调整为数据集大于数据集的 `最大类别id+1`。以COCO数据集为例，查看 `instances_val2017.json` 标注文件，我们可以发现其最大类别id为`90`，因此设置 `num_classes = 91`。

    ```json
    {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ```
    如果您不确定 `num_classes` 需要设置为多少，也可以简单地将其设置为足够大的一个数。（例如，设置`num_classes = 92`或`num_classes = 365`对于COCO数据集都没问题）。
3. 按需调整 [`configs/salience_detr`](configs/salience_detr/) 文件夹下的其他模型参数和 [`train_config.py`](train_config.py) 文件中的训练参数。

## 📥导出ONNX模型

对于想部署我们模型的高级用户，我们提供了脚本来导出ONNX文件。

```shell
python tools/pytorch2onnx.py \
    --model-config /path/to/model.py \
    --checkpoint /path/to/checkpoint.pth \
    --save-file /path/to/save.onnx \
    --simplify \  # 使用onnxsim来简化导出的ONNX文件
    --verify  # 验证导出的ONNX模型和原始pytorch模型的误差
```

请参照 [`tools/pytorch2onnx.py`](tools/pytorch2onnx.py) 文件中的 `ONNXDetector` 类来进行ONNX模型的推理。

## 引用

如果我们的工作对您的研究有帮助，请考虑引用下面论文。

```bibtex
@InProceedings{Hou_2024_CVPR,
    author    = {Hou, Xiuquan and Liu, Meiqin and Zhang, Senlin and Wei, Ping and Chen, Badong},
    title     = {Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17574-17583}
}

@inproceedings{hou2024relation,
  title={Relation DETR: Exploring Explicit Position Relation Prior for Object Detection},
  author={Hou, Xiuquan and Liu, Meiqin and Zhang, Senlin and Wei, Ping and Chen, Badong and Lan, Xuguang},
  booktitle={European conference on computer vision},
  year={2024},
  organization={Springer}
}
```