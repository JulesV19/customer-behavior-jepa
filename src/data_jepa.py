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


def rating_levels(ratings) -> np.ndarray:
    """Notes (demi-étoiles 0.5..5.0) -> niveaux entiers 1..10 (0 réservé = inconnu)."""
    r = np.asarray(ratings, dtype=np.float32)
    return np.rint(r * 2).astype(np.int64)                    # 0.5->1, 1.0->2, ..., 5.0->10


def rating_bias(ratings) -> np.ndarray:
    """Biais user : note - moyenne du user (capture qu'un 4 sévère != un 4 généreux).

    Moyenne calculée sur tout l'historique du user (statistique stable ; la fuite via les
    2 chunks tenus à l'écart est négligeable sur ~100+ notes).
    """
    r = np.asarray(ratings, dtype=np.float32)
    return (r - r.mean()).astype(np.float32)


def to_rating_chunks(ratings, K: int = CHUNK_SIZE):
    """Notes d'un user -> (levels (M,K) int64, bias (M,K) float32), alignés sur to_chunks."""
    lv = rating_levels(ratings)
    bs = rating_bias(ratings)
    M = len(lv) // K
    if M == 0:
        return np.zeros((0, K), dtype=np.int64), np.zeros((0, K), dtype=np.float32)
    levels = lv[: M * K].reshape(M, K)                        # int64
    bias = bs[: M * K].reshape(M, K).astype(np.float32)       # float32 (pas de troncature)
    return levels, bias


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
        for items, ratings in zip(sequences_df["items"].values, sequences_df["ratings"].values):
            ch = to_chunks(items, K)
            if len(ch) >= min_chunks:
                lv, bs = to_rating_chunks(ratings, K)
                # exclut val (M-2) et test (M-1) sur les 3 tableaux alignés
                self.data.append((ch[:-2], lv[:-2], bs[:-2]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):                     # (items, levels, bias), chacun (M', K)
        return self.data[i]


def collate_train(batch, pad_item: int = 0):
    """Pad la dimension chunk au max du lot. Renvoie (chunks, chunk_mask, levels, bias)."""
    maxM = max(x[0].shape[0] for x in batch)
    K = batch[0][0].shape[1]
    B = len(batch)
    chunks = np.full((B, maxM, K), pad_item, dtype=np.int64)
    levels = np.zeros((B, maxM, K), dtype=np.int64)           # 0 = note inconnue (padding)
    bias = np.zeros((B, maxM, K), dtype=np.float32)
    mask = np.zeros((B, maxM), dtype=bool)
    for i, (it, lv, bs) in enumerate(batch):
        m = it.shape[0]
        chunks[i, :m] = it
        levels[i, :m] = lv
        bias[i, :m] = bs
        mask[i, :m] = True
    return (torch.from_numpy(chunks), torch.from_numpy(mask),
            torch.from_numpy(levels), torch.from_numpy(bias))


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
        # context = items du contexte (rétro-compat : utilisé tel quel par retrieval/knn) ;
        # context_lv/context_bs = notes du contexte (pour l'encodeur online) ;
        # target = chunk cible en items seuls (la cible est RATING-FREE).
        self.context, self.context_lv, self.context_bs, self.target = [], [], [], []
        for items, ratings in zip(sequences_df["items"].values, sequences_df["ratings"].values):
            ch = to_chunks(items, K)
            if len(ch) < min_chunks:
                continue
            lv, bs = to_rating_chunks(ratings, K)
            cut = -1 if split == "test" else -2          # cible = M-1 (test) ou M-2 (val)
            self.context.append(ch[:cut]); self.target.append(ch[cut])
            self.context_lv.append(lv[:cut]); self.context_bs.append(bs[:cut])

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.target[i], self.context_lv[i], self.context_bs[i]


def collate_eval(batch, pad_item: int = 0):
    """Renvoie (chunks_contexte, chunk_mask, target_chunk, levels, bias)."""
    contexts = [b[0] for b in batch]
    targets = np.stack([b[1] for b in batch])        # (B, K)
    maxM = max(x.shape[0] for x in contexts)
    K = targets.shape[1]
    B = len(batch)
    chunks = np.full((B, maxM, K), pad_item, dtype=np.int64)
    levels = np.zeros((B, maxM, K), dtype=np.int64)
    bias = np.zeros((B, maxM, K), dtype=np.float32)
    mask = np.zeros((B, maxM), dtype=bool)
    for i, (x, lv, bs) in enumerate(zip(contexts, [b[2] for b in batch], [b[3] for b in batch])):
        m = x.shape[0]
        chunks[i, :m] = x
        levels[i, :m] = lv
        bias[i, :m] = bs
        mask[i, :m] = True
    return (torch.from_numpy(chunks), torch.from_numpy(mask), torch.from_numpy(targets),
            torch.from_numpy(levels), torch.from_numpy(bias))
