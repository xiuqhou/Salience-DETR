English | [简体中文](README_cn.md)

**Salience DETR**
===

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/salience-detr-enhancing-detection-transformer-1/object-detection-on-coco-2017-val)](https://paperswithcode.com/sota/object-detection-on-coco-2017-val?p=salience-detr-enhancing-detection-transformer-1)
[![arXiv](https://img.shields.io/badge/arXiv-2403.16131-b31b1b.svg)](https://arxiv.org/abs/2403.16131)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![GitHub license](https://img.shields.io/github/license/xiuqhou/Salience-DETR.svg?color=blue)](https://github.com/xiuqhou/Salience-DETR/blob/master/LICENSE)
![GitHub stars](https://img.shields.io/github/stars/xiuqhou/Salience-DETR)
![GitHub forks](https://img.shields.io/github/forks/xiuqhou/Salience-DETR)

This repository is an official implementation of the [Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html) accepeted to **CVPR 2024** (score **553**). Authors: [Xiuquan Hou](https://github.com/xiuqhou), [Meiqin Liu](https://scholar.google.com/citations?user=T07OWMkAAAAJ&hl=zh-CN&oi=ao), Senlin Zhang, [Ping Wei](https://scholar.google.com/citations?user=1OQBtdcAAAAJ&hl=zh-CN&oi=ao), [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).

💖 If our Salience-DETR is helpful to your researches or projects, please star this repository. Thanks! 🤗

<div align="center">
    <img src="images/Salience-DETR.svg">
</div>

<details>

<summary>✨Highlights</summary>

1. We offer a deepened analysis for [scale bias and query redundancy](#id_1) issues of two-stage DETR-like methods.
2. We present a hierarchical filtering mechanism to reduce the computational complexity under salience supervision. The proposed salience supervision benefits to capture [fine-grained object contours](#id_2) even with bounding box annotations.
3. Salience DETR achieves **+4.0%**, **+0.2%**, and **+4.4%** AP on three challenging defect detection tasks, and comparable performance (**49.2** AP) with about only **70\%** FLOPs on COCO 2017.

</details>

<details>

<summary>🔎Visualization</summary>

- Queries in the two-stage selection of existing DETR-like methods is usually **redundant** and have **scale bias** (left).
- **Salience supervision** benefits to capture **object contours** even with only bounding box annotations, for both defect detection and object detection tasks (right).

<h3 align="center">
    <a id="id_1"><img src="images/query_visualization.svg" width="335"></a>
    <a id="id_2"><img src="images/salience_visualization.svg" width="462"></a>
</h3>

</details>

## Update

- [2024-07-18] We release [Relation-DETR](https://github.com/xiuqhou/Relation-DETR), a general and strong object detection model that achieves ***40+% AP using only 2 epochs*** and suppresses most SOTA methods including [DDQ-DETR](https://github.com/jshilong/DDQ/tree/ddq_detr), [StableDINO](https://github.com/idea-research/stable-dino), [Rank-DETR](https://github.com/LeapLabTHU/Rank-DETR), [MS-DETR](https://github.com/Atten4Vis/MS-DETR). Code and checkpoints are available [here](https://github.com/xiuqhou/Relation-DETR).

- [2024-04-19] Salience DETR with [FocalNet-Large](https://github.com/microsoft/FocalNet) achieves **56.8 AP** on COCO val2017, [**config**](configs/salience_detr/salience_detr_focalnet_large_lrf_800_1333.py) and [**checkpoint**](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_focalnet_large_lrf_800_1333_coco_1x.pth) are available!

- [2024-04-08] Update [**config**](configs/salience_detr/salience_detr_convnext_l_800_1333.py) and [**checkpoint**](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_convnext_l_800_1333_coco_1x.pth) of Salience DETR with ConvNeXt-L backbone trained on COCO 2017 (12epoch).

- [2024-04-01] Our Salience DETR with Swin-L backbone achieves **56.5** AP on COCO 2017 (12epoch). The model [**config**](configs/salience_detr/salience_detr_swin_l_800_1333.py) and [**checkpoint**](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_swin_l_800_1333_coco_1x.pth) are available.

- [2024-03-26] We release code of Salience DETR and pretrained weights on COCO 2017 for Salience DETR with ResNet50 backbone.

- [2024-02-29] Salience DETR is accepted in CVPR2024, and code will be released in the repo. Welcome to your attention!

## Model Zoo

### 12 epoch setting

| Model         | backbone                |  mAP  | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |                                                                                                       Download                                                                                                       |
| ------------- | ----------------------- | :---: | :-------: | :-------: | :------: | :------: | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Salience DETR | ResNet50                | 50.0  |   67.7    |   54.2    |   33.3   |   54.4   |   64.4   |           [config](configs/salience_detr/salience_detr_resnet50_800_1333.py) / [checkpoint](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_resnet50_800_1333_coco_1x.pth)           |
| Salience DETR | ConvNeXt-L              | 54.2  |   72.4    |   59.1    |   38.8   |   58.3   |   69.6   |         [config](configs/salience_detr/salience_detr_convnext_l_800_1333.py) / [checkpoint](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_convnext_l_800_1333_coco_1x.pth)         |
| Salience DETR | Swin-L<sub>(IN-22K)     | 56.5  |   75.0    |   61.5    |   40.2   |   61.2   |   72.8   |             [config](configs/salience_detr/salience_detr_swin_l_800_1333.py) / [checkpoint](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_swin_l_800_1333_coco_1x.pth)             |
| Salience DETR | FocalNet-L<sub>(IN-22K) | 57.3  |   75.5    |   62.3    |   40.9   |   61.8   |   74.5   | [config](configs/salience_detr/salience_detr_focalnet_large_lrf_800_1333.py) / [checkpoint](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_focalnet_large_lrf_800_1333_coco_1x.pth) |

### 24 epoch setting

| Model         | backbone |  mAP  | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |                                                                                             Download                                                                                             |
| ------------- | -------- | :---: | :-------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Salience DETR | ResNet50 | 51.2  |   68.9    |   55.7    |   33.9   |   55.5   |   65.6   | [config](configs/salience_detr/salience_detr_resnet50_800_1333.py) / [checkpoint](https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_resnet50_800_1333_coco_2x.pth) |


## 🔧Installation

1. Clone the repository locally:

    ```shell
    git clone https://github.com/xiuqhou/Salience-DETR.git
    cd Salience-DETR/
    ```

2. Create a conda environment and activate it:

    ```shell
    conda create -n salience_detr python=3.8
    conda activate salience_detr
    ```

3. Install PyTorch and Torchvision following the instruction on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). The code requires `python>=3.8, torch>=1.11.0, torchvision>=0.12.0`.

    ```shell
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    ```

4. Install other dependencies with:

    ```shell
    conda install --file requirements.txt -c conda-forge
    ```

That's all, you don't need to compile CUDA operators mannually since we load it automatically when running for the first time.

## 📁Prepare Dataset

Please download [COCO 2017](https://cocodataset.org/) or prepare your own datasets into `data/`, and organize them as following. You can use [`tools/visualize_datasets.py`](tools/visualize_datasets.py) to visualize the dataset annotations to verify its correctness.

```shell
coco/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

<details>

<summary>Example for visualization</summary>

```shell
python tools/visualize_datasets.py \
    --coco-img data/coco/val2017 \
    --coco-ann data/coco/annotations/instances_val2017.json \
    --show-dir visualize_dataset/
```

</details>

## 📚︎Train a model

We use `accelerate` package to natively handle multi GPUs, use `CUDA_VISIBLE_DEVICES` to specify GPU/GPUs. If not specified, the script will use all available GPUs on the node to train.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py    # train with 1 GPU
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py  # train with 2 GPUs
```

Before start training, modify parameters in [`configs/train_config.py`](configs/train_config.py).

<details>

<summary>A simple example for train config</summary>

```python
from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
coco_path = "data/coco"  # /PATH/TO/YOUR/COCODIR
train_transform = presets.detr  # see transforms/presets to choose a transform
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2017.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = None

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
```
</details>

## 📈Evaluation/Test

To evaluate a model with one or more GPUs, specify `CUDA_VISIBLE_DEVICES`, `dataset`, `model` and `checkpoint`.

```shell
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch test.py --coco-path /path/to/coco --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth
```

Optional parameters are as follows, see [test.py](test.py) for full parameters:

- `--show-dir`: path to save detection visualization results.
- `--result`: specify a file to save detection numeric results, end with `.json`.

<details>

<summary>An example for evaluation</summary>

To evaluate `salience_detr_resnet50_800_1333` on `coco` using 8 GPUs, save predictions to `result.json` and visualize results to `visualization/`:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch test.py
    --coco-path data/coco \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint https://github.com/xiuqhou/Salience-DETR/releases/download/v1.0.0/salience_detr_resnet50_800_1333_coco_1x.pth \
    --result result.json \
    --show-dir visualization/
```

</details>

<details>

<summary>Evaluate a json result file</summary>

To evaluate the json result file obtained above, specify the `--result` but not specify `--model`.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py --coco-path /path/to/coco --result /path/to/result.json
```

Optional parameters, see [test.py](test.py) for full parameters:

- `--show-dir`: path to save detection visualization results.

</details>

## ▶︎Inference

Use [`inference.py`](inference.py) to perform inference on images. You should specify the image directory using `--image-dir`.

```shell
python inference.py --image-dir /path/to/images --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth --show-dir /path/to/dir
```

<details>

<summary>An example for inference on an image folder</summary>

To performa inference for images under `images/` and save visualizations to `visualization/`:

```shell
python inference.py \
    --image-dir images/ \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint checkpoint.pth \
    --show-dir visualization/
```

</details>

See [`inference.ipynb`](inference.ipynb) for inference on single image and visualization.

## 🔁Benchmark a model

To test the inference speed, memory cost and parameters of a model, use `tools/benchmark_model.py`.

```shell
python tools/benchmark_model.py --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py
```

## 📍Train your own datasets

To train your own datasets, there are some things to do before training:

1. Prepare your datasets with COCO annotation format, and modify `coco_path` in [`configs/train_config.py`](configs/train_config.py) accordingly.
2. Open model configs under [`configs/salience_detr`](configs/salience_detr) and modify the `num_classes` to a number  larger than `max_category_id + 1` of your dataset. For example, from the following annotation in `instances_val2017.json`, we can find the maximum category_id is `90` for COCO, so we set `num_classes = 91`.

    ```json
    {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ```
    You can simply set `num_classes` to a large enough number if not sure what to set. (For example, `num_classes = 92` or `num_classes = 365` also work for COCO.)
3. If necessary, modify other parameters in model configs under [`configs/salience_detr`](configs/salience_detr/) and [`train_config.py`](train_config.py).

## 📥Export an ONNX model

For advanced users who want to deploy our model, we provide a script to export an ONNX file.

```shell
python tools/pytorch2onnx.py \
    --model-config /path/to/model.py \
    --checkpoint /path/to/checkpoint.pth \
    --save-file /path/to/save.onnx \
    --simplify \  # use onnxsim to simplify the exported onnx file
    --verify  # verify the error between onnx model and pytorch model
```

For inference using the ONNX file, see `ONNXDetector` in [`tools/pytorch2onnx.py`](tools/pytorch2onnx.py)

## Reference

If you find our work helpful for your research, please consider citing:

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