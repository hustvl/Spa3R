<div align="center">

# 🔮 ***[Spae]()ing the Unseen***

### Spa3R: Predictive Spatial Field Modeling for 3D Visual Reasoning

[Haoyi Jiang](https://scholar.google.com/citations?user=_45BVtQAAAAJ)<sup>1</sup>,
Liu Liu<sup>2</sup>, Xinjie Wang<sup>2</sup>, Yonghao He<sup>3</sup>,<br>
Wei Sui<sup>3</sup>, Zhizhong Su<sup>2</sup>,
Wenyu Liu<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1</sup><br>
<sup>1</sup>Huazhong University of Science & Technology,
<sup>2</sup>Horizon Robotics,
<sup>3</sup>D-Robotics

[![arXiv](https://img.shields.io/badge/arXiv-2602.21186-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2602.21186)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/hustvl/Spa3-VLM)
[![License: MIT](https://img.shields.io/github/license/hustvl/Spa3R)](LICENSE)

</div>

## Installation

Please clone this project with `--recursive`.

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

pip install submodules/vggt
pip install -e submodules/lmms-eval
```

## Data Preparation

### 1. Pre-training

We utilize a combination of large-scale indoor scene datasets: [ScanNet](http://www.scan-net.org/) and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/).

### 2. Instruction Tuning

* For video-centric VSI-Bench, we fine-tune on [VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K).
* For image-based benchmarks, we use a composite training set. Please refer to the [VG-LLM datasets](https://github.com/LaVi-Lab/VG-LLM?tab=readme-ov-file#datasets).

## Training

### 1. Spa3R Pre-training

To train the Predictive Spatial Field Modeling (PSFM) framework from scratch:

```
export PYTHONPATH=.
python scripts/train_spa3r.py
```

### 2. Spa3-VLM Instruction Tuning

Set the pre-trained Spa3R path in the script: `geometry_encoder_path=/path/to/spa3r.ckpt`

```
bash scripts/train_vlm_sft.sh
```

## Evaluation

To evaluate Spa3-VLM on spatial reasoning benchmarks:

```
bash scripts/eval_vlm.sh
```

## Citation

If you find our work helpful for your research, please consider starring this repository :star: and citing our work:

```bibtex
@article{Spa3R,
  title={Spa3R: Predictive Spatial Field Modeling for 3D Visual Reasoning},
  author={Haoyi Jiang and Liu Liu and Xinjie Wang and Yonghao He and Wei Sui and Zhizhong Su and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:2602.21186},
  year=2026
}
```


## License

This project is released under the [MIT](LICENSE) License.
