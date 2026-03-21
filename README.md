# DREAM Embed

> **Language-Agnostic Sentence Embeddings** — PyTorch implementation of
> *"Language-Agnostic Representation from Multilingual Sentence Encoders for Cross-Lingual Similarity Estimation"*
> Tiyajamorn et al., EMNLP 2021 · [[Paper]](https://aclanthology.org/2021.emnlp-main.612) · [[Original code]](https://github.com/nattaptiy/qe_disentangled)

[![Tests](https://github.com/YOUR_USERNAME/dream-embed/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/dream-embed/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- [![Demo](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/dream-embed) -->

---

## The problem

Standard multilingual sentence encoders (mBERT, XLM-R, LaBSE) cluster
embeddings **by language, not by meaning**.  If you compute cosine similarity
between an English sentence and its French translation, the result is near 0 —
even though the meaning is identical.

## The solution

DREAM trains a lightweight autoencoder head on top of a frozen backbone.
It splits each embedding into:

```
e  =  ê_M  +  ê_L
      └── meaning  (language-agnostic, used for similarity)
                └── language  (discarded at inference)
```

The head is just **two single-layer MLPs** — the entire trainable component
is tiny (~1.2M parameters for LaBSE backbone).

```
sentence ──▶ [frozen LaBSE] ──▶ raw embedding (768d)
                                        │
                              ┌─────────┴─────────┐
                              │   DREAM head      │
                              │  MLP_M │  MLP_L   │
                              └─────────┬─────────┘
                                   meaning embedding
                                        │
                               cosine similarity  ✓  cross-lingual
```

## Quick start

```bash
pip install -e ".[api,demo]"
```

```python
from dream import DREAMPipeline

pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt")

# Cross-lingual similarity (any language pair)
score = pipe.similarity("The cat sat on the mat.", "Le chat était assis sur le tapis.")
print(score)   # ~0.93

# Batch encode → numpy array (N, 768)
embeddings = pipe.encode(["Hello", "Bonjour", "Hola"])

# N×N similarity matrix
matrix = pipe.similarity_matrix(["Hello", "Bonjour", "Hola", "Ciao"])
```

## Training

```bash
# 1. Download the Tatoeba dataset and split train/val
#    (see notebooks/01_data_preparation.ipynb)

# 2. Train (backbone pre-computes embeddings once, then frees GPU memory)
python scripts/train.py --config configs/train.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/dream_epoch_0010.pt

# Override hyperparameters without editing YAML
python scripts/train.py --epochs 50 --lr 5e-5
```

## Evaluate

```bash
# Reproduce Table 4 from the paper (SemEval-2017 cross-lingual STS)
python scripts/evaluate.py --checkpoint checkpoints/dream_best.pt
```

## Serve

```bash
# Gradio demo (local)
python api/demo.py

# FastAPI (local)
uvicorn api.app:app --reload
# Swagger UI → http://localhost:8000/docs

# Docker
docker build -t dream-embed .
docker run -p 7860:7860 -v $(pwd)/checkpoints:/app/checkpoints dream-embed
```

## Results (reproduced on SemEval-2017 cross-lingual STS)

| Model                          | en-ar           | en-de           | en-tr           | en-es | en-fr           | en-it           | en-nl           | **Avg**   |
| ------------------------------ | --------------- | --------------- | --------------- | ----- | --------------- | --------------- | --------------- | --------------- |
| LaBSE baseline                 | 0.705           | 0.721           | 0.748           | 0.692 | 0.759           | 0.760           | 0.755           | 0.734           |
| **LaBSE + DREAM (ours)** | **0.730** | **0.746** | **0.753** | 0.688 | **0.782** | **0.781** | **0.776** | **0.751** |

## Project structure

```
dream-embed/
├── src/dream/
│   ├── __init__.py         # public API: DREAMPipeline, DREAMModel
│   ├── model.py            # DREAMModel (2× single-layer MLP heads)
│   ├── dataset.py          # LanguagePairDataset, MultilingualDataset
│   ├── loss.py             # L_R + L_M + L_L with LossComponents dataclass
│   ├── trainer.py          # Trainer, EarlyStopping, TrainerConfig
│   └── pipeline.py         # DREAMPipeline — end-to-end inference interface
├── api/
│   ├── app.py              # FastAPI: /embed, /similarity, /similarity_matrix
│   └── demo.py             # Gradio demo (Hugging Face Spaces ready)
├── scripts/
│   ├── train.py            # Training entry point
│   └── evaluate.py         # Pearson correlation on STS2017
├── tests/
│   ├── test_model.py
│   └── test_loss.py
├── configs/train.yaml      # All hyperparameters in one place
├── notebooks/              # Exploratory notebooks (original research)
├── Dockerfile
└── pyproject.toml
```

## Swap the backbone

The design is backbone-agnostic.  To switch from LaBSE to XLM-R-large:

```yaml
# configs/train.yaml
model:
  backbone:      "xlm-roberta-large"
  embedding_dim: 1024          # XLM-R-large output dimension
```

```python
pipe = DREAMPipeline.from_pretrained(
    "checkpoints/dream_xlmr_best.pt",
    backbone_name="xlm-roberta-large",
)
```

## Roadmap

- [ ] WandB integration (3 lines, already stubbed in `trainer.py`)
- [ ] ONNX export for CPU-optimised inference
- [ ] Docker Compose (API + Redis result cache)
- [ ] AWS Lambda / SageMaker deployment guide

## Citation

```bibtex
@inproceedings{tiyajamorn-etal-2021-language,
    title     = "Language-Agnostic Representation from Multilingual Sentence Encoders
                 for Cross-Lingual Similarity Estimation",
    author    = "Tiyajamorn, Nattapong and Kajiwara, Tomoyuki and Arase, Yuki and Onizuka, Makoto",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year      = "2021",
    pages     = "7764--7774",
}
```
