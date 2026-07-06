"""Datasets niveau SÉANCE pour le JEPA d'écoute (lastfm-1K).

Depuis `sessions.parquet` (1 ligne/user : items = idx morceau, sessions = idx de séance),
on reconstruit la suite ORDONNÉE des séances de chaque user, on cape chaque séance, puis :

- **Entraînement** : tuiles NON CHEVAUCHANTES de `WINDOW` séances sur l'historique privé
  des 2 dernières séances (tenues pour val/test). Le Transformer temporel prédit, à chaque
  position d'une tuile, le latent de la séance suivante.
- **Évaluation** (leave-last-session-out) : contexte = jusqu'à `WINDOW` séances précédant la
  cible ; cible = dernière séance (test) ou avant-dernière (val).

Séances de taille variable → padding intra-séance + masque (géré au collate).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

WINDOW = 32          # séances par fenêtre (contexte temporel)
SESSION_CAP = 30     # morceaux max par séance (on garde les plus RÉCENTS)
PAD_IDX = 0


# --------------------------------------------------------------------------- #
# Reconstruction des séances d'un user
# --------------------------------------------------------------------------- #
def split_sessions(items, sessions) -> list[np.ndarray]:
    """(items, idx de séance) -> liste d'arrays, un par séance, dans l'ordre."""
    it = np.asarray(items, dtype=np.int64)
    se = np.asarray(sessions, dtype=np.int64)
    if it.size == 0:
        return []
    bounds = np.flatnonzero(np.diff(se)) + 1                   # sessions est croissant
    return np.split(it, bounds)


def _cap(sess: np.ndarray, cap: int) -> np.ndarray:
    """Cape une séance à ses `cap` derniers morceaux (les plus récents)."""
    return sess[-cap:] if len(sess) > cap else sess


def user_sessions(items, sessions, cap: int = SESSION_CAP) -> list[np.ndarray]:
    """Liste des séances capées d'un user."""
    return [_cap(s, cap) for s in split_sessions(items, sessions)]


# --------------------------------------------------------------------------- #
# Entraînement : tuiles non chevauchantes
# --------------------------------------------------------------------------- #
class JepaTrainDataset(Dataset):
    """Un exemple = une tuile de <= WINDOW séances consécutives (hors 2 dernières séances).

    min_sessions=2 par tuile pour avoir au moins une paire (t -> t+1).
    """

    def __init__(self, sessions_df, window: int = WINDOW, cap: int = SESSION_CAP,
                 min_sessions: int = 2):
        self.data: list[list[np.ndarray]] = []
        for items, sess in zip(sessions_df["items"].values, sessions_df["sessions"].values):
            s_list = user_sessions(items, sess, cap)
            train = s_list[:-2]                                # exclut val (M-2) et test (M-1)
            for start in range(0, len(train), window):
                tile = train[start:start + window]
                if len(tile) >= min_sessions:
                    self.data.append(tile)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):                                  # liste de <=WINDOW arrays (<=cap)
        return self.data[i]


def _pad_tile(tile, maxM, maxL, pad=PAD_IDX):
    """Une tuile (liste d'arrays) -> (tracks (maxM,maxL), tok_mask, sess_mask)."""
    tracks = np.full((maxM, maxL), pad, dtype=np.int64)
    tok_mask = np.zeros((maxM, maxL), dtype=bool)
    sess_mask = np.zeros(maxM, dtype=bool)
    for j, s in enumerate(tile):
        L = len(s)
        tracks[j, :L] = s
        tok_mask[j, :L] = True
        sess_mask[j] = True
    return tracks, tok_mask, sess_mask


def collate_train(batch, pad=PAD_IDX):
    """Pad séances (M) et morceaux (L) au max du lot -> (sessions, tok_mask, sess_mask)."""
    B = len(batch)
    maxM = max(len(t) for t in batch)
    maxL = max(max((len(s) for s in t), default=1) for t in batch)
    tracks = np.full((B, maxM, maxL), pad, dtype=np.int64)
    tok_mask = np.zeros((B, maxM, maxL), dtype=bool)
    sess_mask = np.zeros((B, maxM), dtype=bool)
    for i, tile in enumerate(batch):
        tr, tm, sm = _pad_tile(tile, maxM, maxL, pad)
        tracks[i], tok_mask[i], sess_mask[i] = tr, tm, sm
    return (torch.from_numpy(tracks), torch.from_numpy(tok_mask), torch.from_numpy(sess_mask))


# --------------------------------------------------------------------------- #
# Évaluation : leave-last-session-out
# --------------------------------------------------------------------------- #
class JepaEvalDataset(Dataset):
    """Un user -> (contexte = <=WINDOW séances avant la cible, séance cible tenue à l'écart).

    split='test' : cible = dernière séance (M-1) ; split='val' : avant-dernière (M-2).
    """

    def __init__(self, sessions_df, split: str = "test", window: int = WINDOW,
                 cap: int = SESSION_CAP):
        assert split in {"val", "test"}
        cut = 1 if split == "test" else 2                      # index cible = M - cut
        self.context, self.target = [], []
        for items, sess in zip(sessions_df["items"].values, sessions_df["sessions"].values):
            s_list = user_sessions(items, sess, cap)
            ti = len(s_list) - cut
            if ti < 1:                                         # besoin d'>=1 séance de contexte
                continue
            self.context.append(s_list[max(0, ti - window):ti])
            self.target.append(s_list[ti])

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.target[i]


def collate_eval(batch, pad=PAD_IDX):
    """-> (sessions, tok_mask, sess_mask, target_tracks, target_mask)."""
    B = len(batch)
    contexts = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    maxM = max(len(c) for c in contexts)
    maxL = max(max((len(s) for s in c), default=1) for c in contexts)
    maxLt = max(len(t) for t in targets)

    tracks = np.full((B, maxM, maxL), pad, dtype=np.int64)
    tok_mask = np.zeros((B, maxM, maxL), dtype=bool)
    sess_mask = np.zeros((B, maxM), dtype=bool)
    tgt = np.full((B, maxLt), pad, dtype=np.int64)
    tgt_mask = np.zeros((B, maxLt), dtype=bool)
    for i, (ctx, t) in enumerate(zip(contexts, targets)):
        tr, tm, sm = _pad_tile(ctx, maxM, maxL, pad)
        tracks[i], tok_mask[i], sess_mask[i] = tr, tm, sm
        tgt[i, :len(t)] = t
        tgt_mask[i, :len(t)] = True
    return (torch.from_numpy(tracks), torch.from_numpy(tok_mask), torch.from_numpy(sess_mask),
            torch.from_numpy(tgt), torch.from_numpy(tgt_mask))
