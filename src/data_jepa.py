"""Datasets niveau chunk pour le JEPA de trajectoires.

Découpage : la séquence d'un user (index items 1..N, déjà triée et cappée à 500)
est découpée en chunks NON CHEVAUCHANTS de K films. Le reste (< K) est ignoré,
pour n'avoir que des chunks pleins (pas de padding intra-chunk).

Split leave-last-chunk-out (analogue chunk du leave-one-out) :
- test = dernier chunk (index M-1), contexte = chunks[0 : M-1]
- val  = avant-dernier chunk (M-2), contexte = chunks[0 : M-2]
- train = tout le reste : on prédit chaque chunk t+1 depuis chunks[0..t],
          pour t+1 <= M-3 (donc on entraîne sur chunks[0 : M-2]).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CHUNK_SIZE = 5


def to_chunks(items, K: int = CHUNK_SIZE) -> np.ndarray:
    """Liste d'items -> tableau (M, K) de chunks pleins non chevauchants."""
    M = len(items) // K
    if M == 0:
        return np.zeros((0, K), dtype=np.int64)
    return np.asarray(items[: M * K], dtype=np.int64).reshape(M, K)


# --------------------------------------------------------------------------- #
# Entraînement : prédiction de chaque chunk suivant dans le contexte
# --------------------------------------------------------------------------- #
class JepaTrainDataset(Dataset):
    """Un user -> ses chunks de contexte (on retire les 2 derniers = val/test).

    min_chunks=4 garantit au moins une paire (t -> t+1) après retrait des 2 derniers.
    """

    def __init__(self, sequences_df, K: int = CHUNK_SIZE, min_chunks: int = 4):
        self.K = K
        self.data = []
        for items in sequences_df["items"].values:
            ch = to_chunks(items, K)
            if len(ch) >= min_chunks:
                self.data.append(ch[:-2])         # exclut val (M-2) et test (M-1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> np.ndarray:       # (M', K)
        return self.data[i]


def collate_train(batch, pad_item: int = 0):
    """Pad la dimension chunk au max du lot. Renvoie (chunks, chunk_mask)."""
    maxM = max(x.shape[0] for x in batch)
    K = batch[0].shape[1]
    B = len(batch)
    chunks = np.full((B, maxM, K), pad_item, dtype=np.int64)
    mask = np.zeros((B, maxM), dtype=bool)
    for i, x in enumerate(batch):
        m = x.shape[0]
        chunks[i, :m] = x
        mask[i, :m] = True
    return torch.from_numpy(chunks), torch.from_numpy(mask)


# --------------------------------------------------------------------------- #
# Évaluation : prédire UN chunk cible tenu à l'écart (val ou test)
# --------------------------------------------------------------------------- #
class JepaEvalDataset(Dataset):
    """Un user -> (chunks de contexte, chunk cible tenu à l'écart).

    split='val'  : contexte = chunks[0:M-2], cible = chunk M-2
    split='test' : contexte = chunks[0:M-1], cible = chunk M-1
    """

    def __init__(self, sequences_df, split: str = "test",
                 K: int = CHUNK_SIZE, min_chunks: int = 3):
        assert split in {"val", "test"}
        self.K = K
        self.context, self.target = [], []
        for items in sequences_df["items"].values:
            ch = to_chunks(items, K)
            if len(ch) < min_chunks:
                continue
            if split == "test":
                self.context.append(ch[:-1]); self.target.append(ch[-1])
            else:  # val
                self.context.append(ch[:-2]); self.target.append(ch[-2])

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.target[i]


def collate_eval(batch, pad_item: int = 0):
    """Renvoie (chunks_contexte, chunk_mask, target_chunk)."""
    contexts = [b[0] for b in batch]
    targets = np.stack([b[1] for b in batch])        # (B, K)
    maxM = max(x.shape[0] for x in contexts)
    K = targets.shape[1]
    B = len(batch)
    chunks = np.full((B, maxM, K), pad_item, dtype=np.int64)
    mask = np.zeros((B, maxM), dtype=bool)
    for i, x in enumerate(contexts):
        m = x.shape[0]
        chunks[i, :m] = x
        mask[i, :m] = True
    return torch.from_numpy(chunks), torch.from_numpy(mask), torch.from_numpy(targets)
