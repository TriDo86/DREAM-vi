import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import glob

class SingleTatoebaDataset(Dataset):
    """
    PyTorch Dataset for the Tatoeba parallel corpus (single language pair).

    Each sample returns 4 elements:
        (a) src_id of a synonym pair
        (b) tar_id of a synonym pair
        (c) src_id of a random NON-synonym pair
        (d) tar_id of a random NON-synonym pair

    Use src_lookup / tar_lookup to decode id → text when needed.
    Call shuffle() at the beginning of each epoch to regenerate random pairs.
    """

    def __init__(self, tsv_path: str, shuffle=True, encoder=None) -> None:
        self.path = tsv_path
        data = pd.read_csv(
            tsv_path, sep="\t", header=None,
            names=["src_id", "src", "tar_id", "tar"],
            on_bad_lines="skip"
        )
        data = data.dropna(subset=["src_id", "src", "tar_id", "tar"])

        self.is_encoded = encoder is not None
        if self.is_encoded:
            src_df = data.drop_duplicates("src_id")[["src_id", "src"]]
            src_embeddings = encoder.encode(
                src_df['src'].to_list(),
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=True,
                device = "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.src_id_to_embedding_index = {src_id: idx for idx, src_id in enumerate(src_df['src_id'].tolist())}
            self.src_embedding_vectors = src_embeddings

            tar_df = data.drop_duplicates("tar_id")[["tar_id", "tar"]]
            tar_embeddings = encoder.encode(
                tar_df['tar'].to_list(),
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=True,
                device = "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.tar_id_to_embedding_index = {tar_id: idx for idx, tar_id in enumerate(tar_df['tar_id'].tolist())}
            self.tar_embedding_vectors = tar_embeddings


        else:
            # id → text mapping, used for decoding during inference or encoding
            self.src_lookup: dict[int, str] = (
                data.drop_duplicates("src_id")
                    .set_index("src_id")["src"]
                    .to_dict()
            )
            self.tar_lookup: dict[int, str] = (
                data.drop_duplicates("tar_id")
                    .set_index("tar_id")["tar"]
                    .to_dict()
            )

        # src_id → set of synonym tar_ids, used for conflict detection
        self.synonym_lookup: dict[int, set[int]] = (
            data.groupby("src_id")["tar_id"].apply(set).to_dict()
        )

        # Ground-truth synonym pairs — order is preserved across epochs
        self.synonym_pairs: list[tuple[int, int]] = list(
            zip(data["src_id"].values, data["tar_id"].values)
        )
        if shuffle:
            random.shuffle(self.synonym_pairs)

        # Pre-computed random pairs; regenerated each epoch via shuffle()
        self.random_pairs: list[tuple[int, int]] = self._build_random_pairs()

    # ------------------------------------------------------------------

    def _build_random_pairs(self, max_attempts: int = 10) -> list[tuple[int, int]]:
        """
        Build a list of random pairs guaranteed to contain no synonym pairs.

        Strategy:
            - Shuffle tar_ids, then resolve conflicts via swapping.
            - If a conflict cannot be resolved by swapping, re-shuffle entirely.

        Args:
            max_attempts: maximum number of re-shuffle attempts before raising.

        Returns:
            List of (src_id, tar_id) where no pair is a known synonym.

        Raises:
            RuntimeError: if conflicts cannot be resolved after max_attempts.
        """
        # src_ids are kept in original order to stay aligned with synonym_pairs
        src_ids = [src_id for src_id, _ in self.synonym_pairs]
        tar_ids = [tar_id for _, tar_id in self.synonym_pairs]
        N = len(src_ids)

        random.shuffle(src_ids)

        for attempt in range(1, max_attempts + 1):
            random.shuffle(tar_ids)

            has_unresolved = False

            for i in range(N):
                synonyms_i = self.synonym_lookup.get(src_ids[i], set())

                # No conflict at position i, skip
                if tar_ids[i] not in synonyms_i:
                    continue

                # Conflict detected — find j to swap with
                is_swapped = False
                for j in range(i + 1, N):
                    synonyms_j = self.synonym_lookup.get(src_ids[j], set())
                    if tar_ids[j] not in synonyms_i and tar_ids[i] not in synonyms_j:
                        tar_ids[i], tar_ids[j] = tar_ids[j], tar_ids[i]
                        is_swapped = True
                        break  # stop after first valid swap

                # No valid j found — trigger a full re-shuffle
                if not is_swapped:
                    has_unresolved = True
                    break

            if not has_unresolved:
                return list(zip(src_ids, tar_ids))

        raise RuntimeError(
            f"Could not resolve all synonym conflicts after {max_attempts} attempts. "
            "Dataset may be too small or synonym density too high."
        )

    # ------------------------------------------------------------------

    def shuffle(self) -> None:
        """
        Regenerate random pairs with a new random order.
        Should be called at the start of each epoch to prevent the model
        from memorizing fixed negative patterns.
        """
        random.shuffle(self.synonym_pairs)
        self.random_pairs = self._build_random_pairs()

    def __len__(self) -> int:
        return len(self.synonym_pairs)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        """
        Returns:
            (a_src_id, b_tar_id, c_src_id, d_tar_id) where
            (a, b) is a synonym pair and (c, d) is a non-synonym pair.
        """
        synonym_src_id, synonym_tar_id = self.synonym_pairs[index]
        random_src_id, random_tar_id = self.random_pairs[index]

        if not self.is_encoded:
            return self.src_lookup[synonym_src_id], self.tar_lookup[synonym_tar_id], self.src_lookup[random_src_id], self.tar_lookup[random_tar_id]
        
        synonym_src_embedding_idx = self.src_id_to_embedding_index[synonym_src_id]
        synonym_tar_embedding_idx = self.tar_id_to_embedding_index[synonym_tar_id]
        random_src_embedding_idx  = self.src_id_to_embedding_index[random_src_id]
        random_tar_embedding_idx  = self.tar_id_to_embedding_index[random_tar_id]

        return (self.src_embedding_vectors[synonym_src_embedding_idx], 
                self.tar_embedding_vectors[synonym_tar_embedding_idx],
                self.src_embedding_vectors[random_src_embedding_idx], 
                self.tar_embedding_vectors[random_tar_embedding_idx])
    

class TatoebaDataset(Dataset):
    def __init__(self, folder_path, shuffle=True, encoder=None):
        all_tsv_paths = glob.glob(f'{folder_path}/*tsv')
        self.single_datasets = [SingleTatoebaDataset(tsv_path=tsv_path, shuffle=shuffle, encoder=encoder) for tsv_path in all_tsv_paths]
        self.flat_index = [(dataset_index, data_index) 
                           for dataset_index, single_dataset in  enumerate(self.single_datasets) 
                           for data_index in range(len(single_dataset))]
        random.shuffle(self.flat_index)

    def shuffle(self):
        for dataset in self.single_datasets:
            dataset.shuffle()

        self.flat_index = [(dataset_index, data_index) 
                           for dataset_index, single_dataset in  enumerate(self.single_datasets) 
                           for data_index in range(len(single_dataset))]
        
        random.shuffle(self.flat_index)

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, index):
        dataset_index, data_index = self.flat_index[index]

        # Unpack 4 embeddings from SingleTatoebaDataset
        src, tar, rand_src, rand_tar = self.single_datasets[dataset_index][data_index]

        # src is always English = 0
        src_lang_id = 0

        # trg_lang is dataset_index (file alphabetical)
        # Arabic=1, Dutch=2, French=3, German=4, Italian=5, Spanish=6, Turkish=7
        tar_lang_id = dataset_index + 1

        return src, tar, rand_src, rand_tar, src_lang_id, tar_lang_id