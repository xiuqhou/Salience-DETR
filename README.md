**Salience-DETR**: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement  
===

By [Xiuquan Hou](https://github.com/xiuqhou), [Meiqin Liu](https://scholar.google.com/citations?user=T07OWMkAAAAJ&hl=zh-CN&oi=ao), Senlin Zhang, [Ping Wei](https://scholar.google.com/citations?user=1OQBtdcAAAAJ&hl=zh-CN&oi=ao), [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).

[![Papers with code](https://img.shields.io/endpoint?url=https%3A%2F%2Fpaperswithcode.com%2Fbadge%2Fsalience-detr-enhancing-detection-transformer-1%2Fobject-detection-on-coco-2017-val)](https://paperswithcode.com/sota/object-detection-on-coco-2017-val)
[![arXiv](https://img.shields.io/badge/arXiv-2403.16131-b31b1b.svg)](https://arxiv.org/abs/2403.16131)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 
[![GitHub license](https://img.shields.io/github/license/xiuqhou/Salience-DETR.svg?color=blue)](https://github.com/xiuqhou/Salience-DETR/blob/master/LICENSE)
![GitHub stars](https://img.shields.io/github/stars/xiuqhou/Salience-DETR)

This repository is an official implementation of the Salience-DETR accepeted to **CVPR 2024** (score **553**).

## ✨Highlights: 

<div align="center">
    <img src="images/Salience-DETR.svg">
</div>

1. We offer a deepened analysis for [scale bias and query redundancy](#id_1) issues of two-stage DETR-like methods.
2. We present a query filtering mechanism to reduce the computational complexity by selectively encoding the most informative queries in each encoder layer and each feature level.
3. The proposed salience supervision benefits to capture [fine-grained object contours](#id_2) even with bounding box annotations.
4. Salience-DETR achieves **+4.0%**, **+0.2%**, and **+4.4%** AP on three challenging defect detection tasks, and comparable performance with about only **70\%** FLOPs on COCO 2017.
5. Our model achieves **49.2** AP with a ResNet-50 backbone under 1x training.

# News

`2024-03`: We release code of Salience-DETR and pretrained weights on COCO 2017 for Salience-DETR with ResNet50 backbone.

`2024-02`: Salience-DETR is accepted in CVPR2024, and code will be released in the repo. Welcome to your attention!

## 🔎Visualization:

- Queries in the two-stage selection of existing DETR-like methods is usually **redundant** and have **scale bias** (left).
- **Salience supervision** benefits to capture **object contours** even with only bounding box annotations, for both defect detection and object detection tasks (right).

<h3 align="center">
    <a id="id_1"><img src="images/query_visualization.svg" width="335"></a>
    <a id="id_2"><img src="images/salience_visualization.svg" width="462"></a>
</h3>

## 🔧Installation

The implementation codes are developed and tested under `python=3.10, pytorch=1.12, torchvision=0.13` on `Ubuntu LTS 20.04`. Other versions might also work properly.

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

3. Install PyTorch and Torchvision following the instruction on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). The code requires `python>=3.8, torch>=1.11.0, torchvision>=0.12.0`. For example:
    
    ```shell
    # just an example, should be installed according to your CUDA version
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    ```

4. Install other dependencies with:

    ```shell
    conda install --file requirements.txt -c conda-forge
    ```

That's all, you don't need to compile CUDA operators mannually since we load it automatically when running for the first time.

## 📁Prepare Dataset

Please download [COCO 2017](https://cocodataset.org/) or prepare your own datasets into `data/`, and organize them as following:

```shell
coco/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

You can use [`tools/visualize_datasets.py`](tools/visualize_datasets.py) to visualize the dataset annotations to verify its correctness.

<details>

<summary>A simple example</summary>

```shell
python tools/visualize_datasets.py \
    --coco-img data/coco/val2017 \
    --coco-ann data/coco/annotations/instances_val2017.json \
    --show-dir visualize_dataset/
```

</details>

## 📚︎Train a model

First, open [`configs/train_config.py`](configs/train_config.py) and modify the `coco_path`, `model_path`, `num_epochs` and other necessary parameters for training, 

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

We use `accelerate` package to natively handle multi GPUs, so training and distributed training use the same command. You just need to specify which GPU/GPUs to use through `CUDA_VISIBLE_DEVICES`

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py    # train with 1 GPU
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py  # train with 2 GPUs
```

If not specify `CUDA_VISIBLE_DEVICES`, the script will use all available GPUs on the node to train.

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

to evaluate `salience_detr_resnet50_800_1333` on `coco` using 8 GPUs, save predictions to `result.json` and visualize results to `visualization/`:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch test.py 
    --coco-path data/coco \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-21_29_56/best_ap.pth \
    --result result.json \
    --show-dir visualization/
```

</details>

To evaluate the json result file obtained above, specify the `--result` but not specify `--model`.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py --coco-path /path/to/coco --result /path/to/result.json
```

Optional parameters, see [test.py](test.py) for full parameters:

- `--show-dir`: path to save detection visualization results.

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

If you find our work helpful for your research, please consider citing the following BibTeX entry or give us a star ⭐.

```bibtex
@inproceedings{hou2024salience,
  title={Salience-DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement},
  author={Hou, Xiuquan and Liu, Meiqin and Zhang, Senlin and Wei, Ping and Chen, Badong},
  booktitle={CVPR},
  year={2024}
}
```