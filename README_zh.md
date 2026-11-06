<div align="center">
<h1>Retain-Neutral Surrogates for Min-Max Unlearning <img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS%202026-68448B?style=flat-square" height="22" /></h1>
<p><i>官方 PyTorch 实现</i></p>
<p><a href="https://huggingface.co/ResearchBrother/machine-unlearning"><img alt="Hugging Face Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=flat-square&labelColor=FFF8DB" height="22" /></a> <img alt="PyTorch 2.6" src="https://img.shields.io/badge/PyTorch%202.6-FFF0EC?style=flat-square&logo=pytorch&logoColor=EE4C2C" height="22" /> <img alt="Python 3.12" src="https://img.shields.io/badge/Python%203.12-EAF2FB?style=flat-square&logo=python&logoColor=3776AB" height="22" /> <img alt="CUDA 12.4" src="https://img.shields.io/badge/CUDA%2012.4-EEF7E0?style=flat-square&logo=nvidia&logoColor=76B900" height="22" /></p>
<p><a href="README_ko.md">한국어</a> | <a href="README.md">English</a> | <b>简体中文</b></p>
<h3>☀️ 感谢来访，祝您今天心情愉快！☀️</h3>
</div>

## 安装

```bash
git clone https://github.com/unlearn-ml/machine-unlearning.git
cd machine-unlearning
pip install -r requirements.txt
pip install .
```

## 使用方法

**1. 遗忘**：用 ROSU 让模型遗忘训练集中随机抽取的 10% 数据

```bash
mu unlearn --ratio 0.1 --seed 42 --output rosu.json
```

**2. 评估**：评估去掉这 10% 数据后重新训练的模型

```bash
mu evaluate --ratio 0.1 --seed 42 --output retrain.json
```

> **注意：** 首次运行时会自动下载 CIFAR-100 数据集，以及我们发布在 [🤗 Hugging Face](https://huggingface.co/ResearchBrother/machine-unlearning) 上的模型。

## 引用

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
