"""
Dataset classes for DREAM training on Tatoeba parallel corpora.

Architecture
------------
Backbone embeddings are pre-computed ONCE when the dataset is constructed and
stored as float tensors in CPU memory.  The DataLoader then returns raw tensors
— no backbone inference happens during training.

This is the design because:
  1. The backbone is frozen; running it every epoch wastes compute.
  2. Training speed is limited by the tiny MLP heads, not by I/O.

Classes
-------
LanguagePairDataset
    Wraps a single TSV file (one language pair).
    Generates synonym pairs (a, b) and random negative pairs (c, d) per epoch.

MultilingualDataset
    Combines multiple LanguagePairDataset instances.
    Assigns language IDs from a configurable mapping.
    Interleaves samples across language pairs.

Usage
-----
    from dream.dataset import MultilingualDataset
    from dream.backbone_factory import create_backbone

    backbone = create_backbone("sentence-transformers/LaBSE")
    ds = MultilingualDataset("data/Tatoeba_Train", backbone=backbone)
    # free backbone from GPU memory after pre-computation
    free_backbone(backbone); del backbone

    loader = DataLoader(ds, batch_size=512, num_workers=4, pin_memory=True)
"""

from __future__ import annotations

import gc
import glob
import logging
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .backbone import BackboneBase

logger = logging.getLogger(__name__)

# Default language → integer ID mapping used by the STS Tatoeba task.
# English is always the source language (ID 0).
# Other IDs are assigned alphabetically by language name.
DEFAULT_LANGUAGE_MAP: dict[str, int] = {
    "en": 0,  # English
    "ar": 1,  # Arabic
    "de": 2,  # German
    "es": 3,  # Spanish
    "fr": 4,  # French
    "it": 5,  # Italian
    "nl": 6,  # Dutch
    "tr": 7,  # Turkish
}


# ---------------------------------------------------------------------------
# Single language-pair dataset
# ---------------------------------------------------------------------------

class LanguagePairDataset(Dataset):
    """
    PyTorch Dataset for one language-pair TSV file from Tatoeba.

    TSV schema (no header):  src_id \\t src_text \\t tgt_id \\t tgt_text

    Each sample returns six tensors:
        src_emb       (D,)  — parallel source sentence embedding
        tgt_emb       (D,)  — parallel target sentence embedding
        rand_src_emb  (D,)  — random (non-parallel) source embedding
        rand_tgt_emb  (D,)  — random (non-parallel) target embedding
        src_lang_id   ()    — integer language ID for the source language
        tgt_lang_id   ()    — integer language ID for the target language

    Negative pairs are regenerated each epoch via `shuffle()`.
    The generation algorithm guarantees no accidental synonym leakage.

    Args:
        tsv_path:     Path to the TSV file.
        backbone:     BackboneBase instance used to pre-compute embeddings.
                      Pass None only for unit testing with synthetic data.
        src_lang_id:  Integer ID for the source language.
        tgt_lang_id:  Integer ID for the target language.
        encode_batch: Batch size passed to backbone.encode().
    """

    def __init__(
        self,
        tsv_path: str | Path,
        backbone: Optional[BackboneBase],
        src_lang_id: int,
        tgt_lang_id: int,
        encode_batch: int = 64,
    ) -> None:
        self.tsv_path   = Path(tsv_path)
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

        # --- load raw data ---------------------------------------------------
        data = pd.read_csv(
            self.tsv_path,
            sep="\t",
            header=None,
            names=["src_id", "src", "tgt_id", "tgt"],
            on_bad_lines="skip",
            dtype={"src_id": str, "tgt_id": str},
        ).dropna(subset=["src_id", "src", "tgt_id", "tgt"])

        if len(data) == 0:
            raise ValueError(f"Empty or invalid TSV: {self.tsv_path}")

        logger.info(
            "  %s: %d pairs (src_lang=%d, tgt_lang=%d)",
            self.tsv_path.name,
            len(data),
            src_lang_id,
            tgt_lang_id,
        )

        # --- synonym lookup: src_id → set of known tgt_ids ------------------
        self._synonym_lookup: dict[str, set[str]] = (
            data.groupby("src_id")["tgt_id"].apply(set).to_dict()
        )

        # --- ground-truth synonym pairs (order preserved across epochs) ------
        self._synonym_pairs: list[tuple[str, str]] = list(
            zip(data["src_id"].tolist(), data["tgt_id"].tolist())
        )

        # --- pre-compute embeddings ------------------------------------------
        if backbone is not None:
            src_df = data.drop_duplicates("src_id")[["src_id", "src"]]
            self._src_vectors: Tensor = backbone.encode(src_df["src"].tolist(), batch_size=encode_batch)
            self._src_id2idx: dict[str, int] = {sid: i for i, sid in enumerate(src_df["src_id"].tolist())}

            tgt_df = data.drop_duplicates("tgt_id")[["tgt_id", "tgt"]]
            self._tgt_vectors: Tensor = backbone.encode(tgt_df["tgt"].tolist(), batch_size=encode_batch)
            self._tgt_id2idx: dict[str, int] = {tid: i for i, tid in enumerate(tgt_df["tgt_id"].tolist())}
        else:
            # synthetic / test mode — caller must set _src_vectors etc. manually
            self._src_vectors = torch.empty(0)
            self._tgt_vectors = torch.empty(0)
            self._src_id2idx  = {}
            self._tgt_id2idx  = {}

        # --- build initial random pairs --------------------------------------
        self._random_pairs: list[tuple[str, str]] = self._build_random_pairs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shuffle(self) -> None:
        """
        Regenerate random negative pairs with a new random order.
        Call this at the START of each epoch to prevent the model from
        memorising fixed negative patterns.
        """
        random.shuffle(self._synonym_pairs)
        self._random_pairs = self._build_random_pairs()

    def __len__(self) -> int:
        return len(self._synonym_pairs)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        syn_src_id,  syn_tgt_id  = self._synonym_pairs[index]
        rand_src_id, rand_tgt_id = self._random_pairs[index]

        src_emb      = self._src_vectors[self._src_id2idx[syn_src_id]]
        tgt_emb      = self._tgt_vectors[self._tgt_id2idx[syn_tgt_id]]
        rand_src_emb = self._src_vectors[self._src_id2idx[rand_src_id]]
        rand_tgt_emb = self._tgt_vectors[self._tgt_id2idx[rand_tgt_id]]

        return (
            src_emb,
            tgt_emb,
            rand_src_emb,
            rand_tgt_emb,
            torch.tensor(self.src_lang_id, dtype=torch.long),
            torch.tensor(self.tgt_lang_id, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_random_pairs(self, max_attempts: int = 10) -> list[tuple[str, str]]:
        """
        Build a list of random negative pairs guaranteed to contain no known
        synonym pairs.

        Strategy: shuffle tgt_ids, resolve conflicts by swapping.
        If a conflict cannot be resolved by swapping, re-shuffle entirely.

        Raises:
            RuntimeError: if conflicts remain after `max_attempts` re-shuffles.
        """
        src_ids = [s for s, _ in self._synonym_pairs]
        tgt_ids = [t for _, t in self._synonym_pairs]
        N = len(src_ids)

        random.shuffle(src_ids)

        for attempt in range(1, max_attempts + 1):
            random.shuffle(tgt_ids)
            has_unresolved = False

            for i in range(N):
                synonyms_i = self._synonym_lookup.get(src_ids[i], set())
                if tgt_ids[i] not in synonyms_i:
                    continue

                # Conflict at position i — find a valid swap partner
                swapped = False
                for j in range(i + 1, N):
                    synonyms_j = self._synonym_lookup.get(src_ids[j], set())
                    if tgt_ids[j] not in synonyms_i and tgt_ids[i] not in synonyms_j:
                        tgt_ids[i], tgt_ids[j] = tgt_ids[j], tgt_ids[i]
                        swapped = True
                        break

                if not swapped:
                    has_unresolved = True
                    break

            if not has_unresolved:
                return list(zip(src_ids, tgt_ids))

        raise RuntimeError(
            f"Could not resolve synonym conflicts in {self.tsv_path.name} "
            f"after {max_attempts} attempts. "
            "The dataset may be too small or synonym density too high."
        )


# ---------------------------------------------------------------------------
# Multilingual dataset
# ---------------------------------------------------------------------------

class MultilingualDataset(Dataset):
    """
    Combines multiple LanguagePairDataset instances into one flat dataset.

    Language IDs are assigned from `language_map`, which defaults to
    DEFAULT_LANGUAGE_MAP.  TSV filenames must contain the target language code
    (e.g. ``tatoeba_en_de.tsv`` → ``de`` → tgt_lang_id=2).

    Call `shuffle()` at the start of each epoch to regenerate negatives for
    all sub-datasets and re-interleave the flat index.

    Args:
        data_dir:     Directory containing ``*.tsv`` files.
        backbone:     BackboneBase instance for pre-computing embeddings.
        language_map: Mapping from ISO-639-1 language code to integer ID.
                      Source language is always ``"en"`` (ID 0).
                      Defaults to DEFAULT_LANGUAGE_MAP when not provided.
        encode_batch: Batch size for backbone.encode().
    """

    def __init__(
        self,
        data_dir: str | Path,
        backbone: BackboneBase,
        language_map: Optional[dict[str, int]] = None,
        encode_batch: int = 64,
    ) -> None:
        
        if language_map is None:
            language_map = DEFAULT_LANGUAGE_MAP.copy()

        data_dir = Path(data_dir)
        tsv_paths = sorted(glob.glob(str(data_dir / "*.tsv")))

        if not tsv_paths:
            raise FileNotFoundError(f"No TSV files found in {data_dir}")

        logger.info("Loading %d language pairs from %s …", len(tsv_paths), data_dir)

        self._datasets: list[LanguagePairDataset] = []
        for path in tsv_paths:
            tgt_code = _infer_tgt_lang_code(path)

            if tgt_code not in language_map:
                logger.warning(
                    "  Skipping %s — language code '%s' not in language_map",
                    path,
                    tgt_code,
                )
                continue

            ds = LanguagePairDataset(
                tsv_path=path,
                backbone=backbone,
                src_lang_id=language_map["en"],
                tgt_lang_id=language_map[tgt_code],
                encode_batch=encode_batch,
            )
            self._datasets.append(ds)

        if not self._datasets:
            raise ValueError("No usable language pairs found.")

        self._flat_index: list[tuple[int, int]] = self._build_flat_index()

        logger.info("Total samples: %d", len(self))

    # ------------------------------------------------------------------

    def shuffle(self) -> None:
        """Shuffle negatives in all sub-datasets and re-interleave the flat index."""
        for ds in self._datasets:
            ds.shuffle()
        self._flat_index = self._build_flat_index()

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, index: int):
        ds_idx, sample_idx = self._flat_index[index]
        return self._datasets[ds_idx][sample_idx]

    # ------------------------------------------------------------------

    def _build_flat_index(self) -> list[tuple[int, int]]:
        index = [
            (ds_i, s_i)
            for ds_i, ds in enumerate(self._datasets)
            for s_i in range(len(ds))
        ]
        random.shuffle(index)
        return index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAME_TO_ISO: dict[str, str] = {
    "arabic":     "ar",  # Arabic
    "german":     "de",  # German
    "spanish":    "es",  # Spanish
    "french":     "fr",  # French
    "italian":    "it",  # Italian
    "dutch":      "nl",  # Dutch
    "turkish":    "tr",  # Turkish
    "chinese":    "zh",  # Chinese
    "japanese":   "ja",  # Japanese
    "korean":     "ko",  # Korean
    "portuguese": "pt",  # Portuguese
    "russian":    "ru",  # Russian
    "hindi":      "hi",  # Hindi
    "vietnamese": "vi",  # Vietnamese
    "thai":       "th",  # Thai
    "polish":     "pl",  # Polish
    "swedish":    "sv",  # Swedish
    "norwegian":  "no",  # Norwegian
    "danish":     "da",  # Danish
    "finnish":    "fi",  # Finnish
    "czech":      "cs",  # Czech
    "hungarian":  "hu",  # Hungarian
    "romanian":   "ro",  # Romanian
    "nepali":     "ne",  # Nepali
    "sinhala":    "si",  # Sinhala
    "estonian":   "et",  # Estonian
    "hebrew":     "he",  # Hebrew
    "indonesian": "id",  # Indonesian
    "malay":      "ms",  # Malay
    "persian":    "fa",  # Persian
    "ukrainian":  "uk",  # Ukrainian
    "greek":      "el",  # Greek
    "bulgarian":  "bg",  # Bulgarian
    "catalan":    "ca",  # Catalan
    "serbian":    "sr",  # Serbian
    "slovak":     "sk",  # Slovak
    "slovenian":  "sl",  # Slovenian
    "latvian":    "lv",  # Latvian
    "lithuanian": "lt",  # Lithuanian
    "tagalog":    "tl",  # Tagalog
    "bengali":    "bn",  # Bengali
    "urdu":       "ur",  # Urdu
    "swahili":    "sw",  # Swahili
}


def _infer_tgt_lang_code(tsv_path: str) -> str:
    """
    Infer the target language ISO-639-1 code from a TSV filename.

    Expected filename pattern (Tatoeba web export):
        ``Sentence pairs in English-Arabic - 2026-03-12.tsv``  →  ``"ar"``
        ``Sentence pairs in English-German - 2025-11-01.tsv``  →  ``"de"``

    Raises:
        ValueError: if the filename does not match the expected pattern
                    or the language name is not in the lookup table.
    """
    stem = Path(tsv_path).stem

    m = re.search(r"English-(\w+)", stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern "
            f"'Sentence pairs in English-<Language> - YYYY-MM-DD': {stem!r}"
        )

    lang_word = m.group(1).lower()
    if lang_word not in _NAME_TO_ISO:
        raise ValueError(
            f"Unknown language name {lang_word!r} in filename {stem!r}. "
            f"Add it to _NAME_TO_ISO if needed."
        )

    return _NAME_TO_ISO[lang_word]


def free_backbone(backbone: BackboneBase) -> None:
    """
    Release backbone from memory after pre-computing all embeddings.
    Call this after MultilingualDataset construction to reclaim VRAM.

    Example::

        ds = MultilingualDataset("data/train", backbone=backbone)
        free_backbone(backbone)
        del backbone
    """
    try:
        backbone.to("cpu")
    except Exception:
        pass
    del backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()