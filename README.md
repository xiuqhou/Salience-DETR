**Salience-DETR**: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement  
===

By [Xiuquan Hou](https://github.com/xiuqhou), [Meiqin Liu](https://scholar.google.com/citations?user=T07OWMkAAAAJ&hl=zh-CN&oi=ao), Senlin Zhang, [Ping Wei](https://scholar.google.com/citations?user=1OQBtdcAAAAJ&hl=zh-CN&oi=ao), [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).

<div align="center">
    <img src="https://img.shields.io/github/license/xiuqhou/Salience-DETR.svg?color=blue" alt="license-Apache2.0">
    <img src="https://img.shields.io/github/stars/xiuqhou/Salience-DETR" alt="Github-stars">
</div>

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

## 🔎Visualization:

- Queries in the two-stage selection of existing DETR-like methods is usually **redundant** and have **scale bias** (left).
- **Salience supervision** benefits to capture **object contours** even with only bounding box annotations, for both defect detection and object detection tasks (right).

<h3 align="center">
    <a><img src="images/query_visualization.svg" width="335" id="id_1"></a>
    <a><img src="images/salience_visualization.svg" width="462" id="id_2"></a>
</h3>

## Reference

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{hou2024salience,
  title={Salience-DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement},
  author={Hou, Xiuquan and Liu, Meiqin and Zhang, Senlin and Wei, Ping and Chen, Badong}
  booktitle={CVPR},
  year={2024}
}
```