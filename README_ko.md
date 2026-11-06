<div align="center">
<h1>Retain-Neutral Surrogates for Min-Max Unlearning <img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS%202026-68448B?style=flat-square" height="22" /></h1>
<p><i>공식 PyTorch 구현</i></p>
<p><a href="https://huggingface.co/ResearchBrother/machine-unlearning"><img alt="Hugging Face Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=flat-square&labelColor=FFF8DB" height="22" /></a> <img alt="PyTorch 2.6" src="https://img.shields.io/badge/PyTorch%202.6-FFF0EC?style=flat-square&logo=pytorch&logoColor=EE4C2C" height="22" /> <img alt="Python 3.12" src="https://img.shields.io/badge/Python%203.12-EAF2FB?style=flat-square&logo=python&logoColor=3776AB" height="22" /> <img alt="CUDA 12.4" src="https://img.shields.io/badge/CUDA%2012.4-EEF7E0?style=flat-square&logo=nvidia&logoColor=76B900" height="22" /></p>
<p><b>한국어</b> | <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a></p>
<h3>☀️ 방문해 주셔서 감사합니다. 오늘도 행복한 하루 보내세요! ☀️</h3>
</div>

## 설치

```bash
git clone https://github.com/unlearn-ml/machine-unlearning.git
cd machine-unlearning
pip install -r requirements.txt
pip install .
```

## 사용 방법

**1. 언러닝**: 학습 데이터에서 무작위로 뽑은 10%를 ROSU로 언러닝합니다

```bash
mu unlearn --ratio 0.1 --seed 42 --output rosu.json
```

**2. 평가**: 해당 10%를 제외하고 재학습한 모델을 평가합니다

```bash
mu evaluate --ratio 0.1 --seed 42 --output retrain.json
```

> **참고:** 처음 실행하면 CIFAR-100 데이터셋과 [🤗 Hugging Face](https://huggingface.co/ResearchBrother/machine-unlearning)에 공개된 모델이 자동으로 다운로드됩니다.

## 인용

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
