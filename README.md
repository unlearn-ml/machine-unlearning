<div align="center">
<h1>Retain-Neutral Surrogates for Min-Max Unlearning <img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS%202026-68448B?style=flat-square" height="22" /></h1>
<p><i>Official PyTorch implementation</i></p>
<p><a href="https://huggingface.co/ResearchBrother/machine-unlearning"><img alt="Hugging Face Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=flat-square&labelColor=FFF8DB" height="22" /></a> <img alt="PyTorch 2.6" src="https://img.shields.io/badge/PyTorch%202.6-FFF0EC?style=flat-square&logo=pytorch&logoColor=EE4C2C" height="22" /> <img alt="Python 3.12" src="https://img.shields.io/badge/Python%203.12-EAF2FB?style=flat-square&logo=python&logoColor=3776AB" height="22" /> <img alt="CUDA 12.4" src="https://img.shields.io/badge/CUDA%2012.4-EEF7E0?style=flat-square&logo=nvidia&logoColor=76B900" height="22" /></p>
<p><a href="README_ko.md">한국어</a> | <b>English</b> | <a href="README_zh.md">简体中文</a></p>
<h3>☀️ Thanks for stopping by. Have a great day! ☀️</h3>
</div>

## Installation

```bash
git clone https://github.com/unlearn-ml/machine-unlearning.git
cd machine-unlearning
pip install -r requirements.txt
pip install .
```

## Usage

**1. Unlearn** a random 10% of the training set with ROSU

```bash
mu unlearn --ratio 0.1 --seed 42 --output rosu.json
```

**2. Evaluate** the model retrained without that 10%

```bash
mu evaluate --ratio 0.1 --seed 42 --output retrain.json
```

> **Note:** The first run downloads CIFAR-100 and our models from [🤗 Hugging Face](https://huggingface.co/ResearchBrother/machine-unlearning).

## Citation

```bibtex
@misc{cai2026retainneutralsurrogatesminmaxunlearning,
      title={Retain-Neutral Surrogates for Min-Max Unlearning},
      author={Junhao Cai and Dohun Kim and Dowon Kim and Sung Il Choi and Chengjun Jin and Juhyun Park and Changhee Joo},
      year={2026},
      eprint={2605.05871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.05871},
}
```
