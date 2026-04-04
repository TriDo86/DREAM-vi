# DREAM — Disentangled Representation for Cross-lingual Meaning

A faithful PyTorch reimplementation of **Tiyajamorn et al., EMNLP 2021**:

> [*Language-agnostic Sentence Representations*](https://aclanthology.org/2021.emnlp-main.612.pdf)

The DREAM model splits a frozen multilingual sentence embedding into two disentangled subspaces — a **meaning embedding** that is language-agnostic, and a **language embedding** that captures language-specific surface features.

```
"The cat sat on the mat."        ─┐
"Le chat est sur le tapis."      ─┼─ backbone ─► DREAM ─► meaning ≈ meaning
"Die Katze saß auf der Matte."   ─┘                       language ≠ language
```

---

## Architecture

```
frozen backbone  (LaBSE / BGE-M3 / XLM-R / mBERT)
        │
        │  sentence embedding  e ∈ ℝᵈ
        ▼
  ┌──────────────────────────────────┐
  │           DREAMModel             │
  │   meaning_encoder   (Linear)   ──┼──► ê_M   language-agnostic
  │   language_encoder  (Linear)   ──┼──► ê_L   language-specific
  │   language_identifier (Linear) ──┼──► lang-id logits
  └──────────────────────────────────┘
        │
        └── Loss:  L = L_R + L_M + L_L
```

**Loss terms** (eqs. 2–11 from the paper):

| Term                            | Role                                                       |
| ------------------------------- | ---------------------------------------------------------- |
| **L_R** — Reconstruction | `ê_M + ê_L ≈ e`  (autoencoder constraint)             |
| **L_M** — Meaning        | Push parallel pairs together; push random pairs apart      |
| **L_L** — Language       | Cluster same-language embeddings + language identification |

---

## Results

Pearson correlation on **STS-2017** cross-lingual tracks.
Each backbone is evaluated before and after applying the DREAM head.

| Backbone                      |     en–de     |     en–fr     |     en–es     |       Avg       |
| ----------------------------- | :-------------: | :-------------: | :-------------: | :-------------: |
| mBERT (backbone only)         |      0.650      |      0.701      |      0.723      |      0.691      |
| **mBERT + DREAM**       | **0.712** | **0.758** | **0.781** | **0.750** |
| LaBSE (backbone only)         |      0.812      |      0.843      |      0.861      |      0.839      |
| **LaBSE + DREAM**       | **0.831** | **0.857** | **0.878** | **0.855** |
| XLM-R large (backbone only)   |      0.834      |      0.861      |      0.874      |      0.856      |
| **XLM-R large + DREAM** | **0.849** | **0.873** | **0.889** | **0.870** |

### Embedding Space Analysis via t-SNE

Columns represent three embedding spaces:

- The raw backbone (left)
- The DREAM meaning subspace (center)
- The DREAM language subspace (right).

After disentanglement, meaning embeddings from different languages converge into a shared region, while language embeddings form distinct per-language clusters.

![mBERT t-SNE](assets/bert-base-multilingual-cased_en-vi_tsne.png)

---

## Project Structure

```
dream-embed/
├── assets/
│   ├── bert-base-multilingual-cased_tsne.png
│   ├── LaBSE_tsne.png
│   └── xlm-roberta-large_tsne.png
├── configs/
│   └── train.yaml
├── data/                          # (gitignored) — see "Training | 1 - Prepare data" section below
│   ├── STS17/
│   ├── Tatoeba_Train/
│   └── Tatoeba_Val/
├── checkpoints/                   # (gitignored)
│   ├── labse/
│   ├── mbert/
│   └── xlmr/
├── notebooks/
│   └── evaluation.ipynb           # STS-2017 Pearson eval + t-SNE plots
├── scripts/
│   └── train.py                   # Training entry point (CLI)
├── src/
│   └── dream/
│       ├── __init__.py
│       ├── backbone.py            # BackboneBase abstraction + 3 adapters
│       ├── backbone_factory.py    # create_backbone() entry point
│       ├── dataset.py             # Tatoeba TSV → pre-computed tensor dataset
│       ├── loss.py                # L_R, L_M, L_L loss terms
│       ├── model.py               # DREAMModel (2-head MLP)
│       ├── pipeline.py            # DREAMPipeline — end-to-end inference API
│       └── trainer.py             # Trainer with EarlyStopping + checkpointing
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/yourusername/dream-embed.git
cd dream-embed
pip install -r requirements.txt
```

### Inference

```python
from dream.pipeline import DREAMPipeline

pipe = DREAMPipeline.from_pretrained(
    "checkpoints/dream_best.pt",
    backbone="sentence-transformers/LaBSE",
)

# Cross-lingual similarity
score = pipe.similarity("The cat is on the mat.", "Die Katze saß auf der Matte.")
print(score)  # ~0.88

# Batch encode — returns (N, D) numpy array
embeddings = pipe.encode(["Hello world", "Bonjour le monde", "Hola mundo"])
print(embeddings.shape)  # (3, 768)

# Similarity matrix
matrix = pipe.similarity_matrix(["Hello", "Bonjour", "Hola"])
print(matrix)  # (3, 3) cosine similarity
```

---

## Training
### 1 — Prepare data

**Step 1 — Download data**

Go to [Tatoeba](https://tatoeba.org/en/downloads), download the sentence pairs for each language pair you want, and place all TSV files into `data/Tatoeba/`:

```
data/
└── Tatoeba/
    ├── Sentence pairs in English-Arabic - 2026-03-12.tsv
    ├── Sentence pairs in English-Dutch - 2026-03-12.tsv
    ├── Sentence pairs in English-French - 2026-03-12.tsv
    ├── Sentence pairs in English-German - 2026-03-12.tsv
    ├── Sentence pairs in English-Italian - 2026-03-12.tsv
    ├── Sentence pairs in English-Spanish - 2026-03-12.tsv
    └── Sentence pairs in English-Turkish - 2026-03-12.tsv
```

> The original paper uses the 7 language pairs above. You can add more — see the note on adding new languages below.

**Step 2 — Split into train/val**

Run `scripts/split_data.py` to split each TSV into training and validation sets:

```bash
# Default: 90% train / 10% val, reads from data/Tatoeba/, writes to data/Tatoeba_Train/ and data/Tatoeba_Val/
python scripts/split_data.py

# Custom split ratio and directories
python scripts/split_data.py --src data/Tatoeba --train data/Tatoeba_Train --val data/Tatoeba_Val --val-ratio 0.1 --seed 86
```

After this step your `data/` directory should look like:

```
data/
├── Tatoeba_Train/
│   ├── Sentence pairs in English-Arabic - 2026-03-12.tsv
│   └── ...
└── Tatoeba_Val/
    ├── Sentence pairs in English-Arabic - 2026-03-12.tsv
    └── ...
```

Each TSV file is tab-separated with no header: `src_id`, `src_text`, `tgt_id`, `tgt_text`.

**Adding a new language pair**

The source language is hardcoded as English. To add a new `English-XXX` pair, open `src/dream/dataset.py` and update two dictionaries:

1. `_NAME_TO_ISO` — maps the language name as it appears in the Tatoeba filename to its ISO-639-1 code:
```python
_NAME_TO_ISO: dict[str, str] = {
    ...
    "vietnamese": "vi",   # add the name exactly as it appears in the filename
}
```

2. `DEFAULT_LANGUAGE_MAP` — assigns a unique integer ID to each language (used as the classification label by the model):
```python
DEFAULT_LANGUAGE_MAP: dict[str, int] = {
    ...
    "vi": 8,   # assign the next available integer ID
}
```

> If your source language is not English, you will need to modify the dataset code manually.

### 2 — Configure
Edit `configs/train.yaml` to set backbone, embedding dimension, batch size, and device.

### 3 — Train
```bash
# Basic run
python scripts/train.py

# Override config values
python scripts/train.py --epochs 50 --lr 5e-5 --device cuda

# Resume from checkpoint
python scripts/train.py --resume checkpoints/dream_epoch_0010.pt
```

---

## Supported Backbones

| Model                                        | `backbone_type` | `embedding_dim` | Notes                             |
| -------------------------------------------- | :---------------: | :---------------: | --------------------------------- |
| `sentence-transformers/LaBSE`              |      `st`      |        768        | Best overall multilingual quality |
| `BAAI/bge-m3`                              |      `bge`      |       1024       | Strong on Asian languages         |
| `FacebookAI/xlm-roberta-large`             |      `hf`      |       1024       | Largest; best absolute quality    |
| `google-bert/bert-base-multilingual-cased` |      `hf`      |        768        | Lightest; fastest training        |

Any model loadable via `sentence-transformers`, `FlagEmbedding`, or HuggingFace `AutoModel` is supported through the `BackboneBase` abstraction — no changes to the training code required.

To use a different backbone, implement the `BackboneBase` interface defined in `src/dream/backbone.py`, then pass your instance directly to `create_backbone()` — it will be returned as-is.

---

## Key Design Decisions

**Frozen backbone + pre-computed embeddings.** The backbone runs once before training begins. Only the three Linear layers of DREAMModel are trained, making each epoch very fast regardless of backbone size.

**`BackboneBase` abstraction.** A unified `.encode()` contract hides the incompatible APIs of sentence-transformers, FlagEmbedding, and raw HuggingFace AutoModel. The rest of the codebase never imports adapter classes directly — only `create_backbone()`.

**Synonym-safe negatives.** `_build_random_pairs()` guarantees no accidental synonym leakage in random negative pairs via swap-based conflict resolution, regenerated every epoch.

---

## Evaluation

Open `notebooks/evaluation.ipynb`, set `BACKBONE_NAME` and `CHECKPOINT_PATH` at the top, then run all cells. The notebook reproduces Table 4 from the paper: Pearson correlation on STS-2017 cross-lingual tracks and t-SNE visualizations of meaning vs. language embeddings.

---

## Reference

```bibtex
@inproceedings{tiyajamorn-etal-2021-language,
    title = "Language-agnostic Representation from Multilingual Sentence Encoders for Cross-lingual Similarity Estimation",
    author = "Tiyajamorn, Nattapong  and
      Kajiwara, Tomoyuki  and
      Arase, Yuki  and
      Onizuka, Makoto",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.612",
    pages = "7764--7774",
    abstract = "We propose a method to distill a language-agnostic meaning embedding from a multilingual sentence encoder. By removing language-specific information from the original embedding, we retrieve an embedding that fully represents the sentence{'}s meaning. The proposed method relies only on parallel corpora without any human annotations. Our meaning embedding allows efficient cross-lingual sentence similarity estimation by simple cosine similarity calculation. Experimental results on both quality estimation of machine translation and cross-lingual semantic textual similarity tasks reveal that our method consistently outperforms the strong baselines using the original multilingual embedding. Our method consistently improves the performance of any pre-trained multilingual sentence encoder, even in low-resource language pairs where only tens of thousands of parallel sentence pairs are available.",
}
```

---

## License

MIT
