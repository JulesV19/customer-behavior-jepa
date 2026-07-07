"""Datasets pour la prédiction de NOTES masquées (tâche pivot 2026-07-07).

Un user = un SAC de films notés. On montre au modèle des films notés (contexte), on en
masque d'autres (identité + genome visibles, note cachée) et il doit deviner la note.

Cible = RÉSIDU `note − μ − b_i − b_u` (biais retirés analytiquement, cf. la décomposition
de Koren et `src/rating_baselines.py`). μ et b_i sont GLOBAUX (estimés une fois sur le
train). b_u est estimé par user sur les notes VISIBLES seulement (jamais les masquées =
pas de fuite). Le modèle apprend le terme d'interaction ; on rajoute les biais pour la RMSE.

Split sans fuite (`split_user_items`) : par user, une fraction fixe et reproductible de ses
notes est tenue en test et n'apparaît JAMAIS à l'entraînement. Ce même split sert au mur
baseline (évaluation strictement comparable).

Convention d'items (cf. data_prep) : index interne 1..n_items, 0 = padding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .rating_baselines import fit_bias, load_long

RATING_MIN, RATING_MAX = 0.5, 5.0


# --------------------------------------------------------------------------- #
# Split reproductible, par user (identique pour modèle ET baseline)
# --------------------------------------------------------------------------- #
def split_user_items(sequences_df: pd.DataFrame, frac: float = 0.2, seed: int = 0,
                     min_ratings: int = 10) -> list[np.ndarray]:
    """Par user, masque booléen aligné sur `items` : True = note tenue en test.

    RNG dérivé de (seed, userId) → déterministe et indépendant de l'ordre des lignes.
    Users trop courts (< min_ratings) : entièrement en train (masque tout False).
    """
    masks = []
    for uid, items in zip(sequences_df["userId"].values, sequences_df["items"].values):
        n = len(items)
        m = np.zeros(n, dtype=bool)
        if n >= min_ratings:
            rng = np.random.default_rng([seed, int(uid)])
            k = int(round(n * frac))
            if k > 0:
                m[rng.choice(n, size=k, replace=False)] = True
        masks.append(m)
    return masks


def carve_val(sequences_df: pd.DataFrame, test_masks: list[np.ndarray],
              frac: float = 0.1, seed: int = 7, min_pool: int = 10) -> list[np.ndarray]:
    """Prélève un masque de VALIDATION dans le pool restant (hors test), disjoint du test.

    Sert au suivi par epoch sans toucher au test (le juge final). RNG dérivé de (seed, uid).
    """
    vals = []
    for uid, items, tm in zip(sequences_df["userId"].values,
                              sequences_df["items"].values, test_masks):
        n = len(items)
        vm = np.zeros(n, dtype=bool)
        pool = np.flatnonzero(~tm)
        if len(pool) >= min_pool:
            rng = np.random.default_rng([seed, int(uid)])
            k = int(round(len(pool) * frac))
            if k > 0:
                vm[rng.choice(pool, size=k, replace=False)] = True
        vals.append(vm)
    return vals


def union_masks(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """OR élément-par-élément de deux listes de masques (test ∪ val = held-out train)."""
    return [x | y for x, y in zip(a, b)]


def global_biases(sequences_df: pd.DataFrame, test_masks: list[np.ndarray],
                  n_items: int, lam_i: float = 5.0):
    """μ et b_i estimés sur les notes de TRAIN seulement (test exclu via test_masks).

    Renvoie (mu: float, b_i: np.ndarray de taille n_items+1, 0 = inconnu/padding).
    """
    users, items, ratings = [], [], []
    for uid, it, rt, tm in zip(sequences_df["userId"].values, sequences_df["items"].values,
                               sequences_df["ratings"].values, test_masks):
        keep = ~tm
        it = np.asarray(it)[keep]; rt = np.asarray(rt)[keep]
        users.append(np.full(len(it), uid)); items.append(it); ratings.append(rt)
    train_long = pd.DataFrame({"user": np.concatenate(users),
                               "item": np.concatenate(items),
                               "rating": np.concatenate(ratings).astype(np.float32)})
    mu, b_i, _ = fit_bias(train_long, lam_i=lam_i, lam_u=lam_i)
    b_i_arr = np.zeros(n_items + 1, dtype=np.float32)
    b_i_arr[b_i.index.values] = b_i.values.astype(np.float32)
    return float(mu), b_i_arr


def _rating_level(r: np.ndarray) -> np.ndarray:
    """Note demi-étoile 0.5..5.0 -> niveau 1..10 (0 = inconnu, cf. ItemTokenizer)."""
    return np.rint(np.asarray(r, dtype=np.float32) * 2).astype(np.int64)


# --------------------------------------------------------------------------- #
# Entraînement : masquage ré-échantillonné à chaque epoch
# --------------------------------------------------------------------------- #
class RatingTrainDataset(Dataset):
    """Un user -> son POOL de train (test exclu) ; masquage aléatoire ~mask_frac / accès.

    Chaque __getitem__ re-tire un masque (augmentation, masques frais à chaque epoch).
    Fournit résidu-cible et features de note aux positions VISIBLES uniquement.
    """

    def __init__(self, sequences_df, test_masks, mu, b_i_arr,
                 mask_frac: float = 0.2, min_pool: int = 5):
        self.mu = mu
        self.b_i = b_i_arr
        self.mask_frac = mask_frac
        self.pool_items, self.pool_ratings = [], []
        for it, rt, tm in zip(sequences_df["items"].values,
                              sequences_df["ratings"].values, test_masks):
            keep = ~tm
            items = np.asarray(it, dtype=np.int64)[keep]
            ratings = np.asarray(rt, dtype=np.float32)[keep]
            if len(items) >= min_pool:
                self.pool_items.append(items)
                self.pool_ratings.append(ratings)

    def __len__(self):
        return len(self.pool_items)

    def __getitem__(self, i):
        items = self.pool_items[i]
        ratings = self.pool_ratings[i]
        n = len(items)
        k = max(1, int(round(n * self.mask_frac)))
        # default_rng() sans graine = entropie fraîche par appel -> masques distincts même
        # entre workers DataLoader (le RNG global numpy se duplique entre workers).
        masked = np.zeros(n, dtype=bool)
        masked[np.random.default_rng().choice(n, size=k, replace=False)] = True
        return self._build(items, ratings, masked)

    def _build(self, items, ratings, masked):
        """Assemble un exemple (visibles avec note, masqués rating-free + résidu cible)."""
        vis = ~masked
        # b_u estimé sur les VISIBLES seulement (pas de fuite)
        resid_vis = ratings[vis] - self.mu - self.b_i[items[vis]]
        b_u = float(resid_vis.mean()) if vis.any() else 0.0

        levels = np.zeros(len(items), dtype=np.int64)
        bias = np.zeros(len(items), dtype=np.float32)
        levels[vis] = _rating_level(ratings[vis])
        # feature "biais user" du tokenizer = écart à la moyenne des notes visibles
        bias[vis] = ratings[vis] - float(ratings[vis].mean()) if vis.any() else 0.0

        target_resid = np.zeros(len(items), dtype=np.float32)
        target_resid[masked] = ratings[masked] - self.mu - self.b_i[items[masked]] - b_u

        return {
            "items": torch.from_numpy(items),
            "levels": torch.from_numpy(levels),
            "bias": torch.from_numpy(bias),
            "target_mask": torch.from_numpy(masked),
            "target_resid": torch.from_numpy(target_resid),
        }


# --------------------------------------------------------------------------- #
# Évaluation : contexte = tout le pool train ; cibles = notes TEST tenues à l'écart
# --------------------------------------------------------------------------- #
class RatingEvalDataset(Dataset):
    """Un user -> (contexte train tout visible) + (films test rating-free à prédire).

    Renvoie aussi la NOTE vraie et la prédiction-baseline (μ+b_i+b_u) par cible, pour
    reconstruire la note prédite = baseline + résidu et calculer la RMSE.
    """

    def __init__(self, sequences_df, test_masks, mu, b_i_arr, min_pool: int = 5):
        self.mu = mu
        self.b_i = b_i_arr
        self.samples = []
        for it, rt, tm in zip(sequences_df["items"].values,
                              sequences_df["ratings"].values, test_masks):
            if not tm.any():
                continue
            it = np.asarray(it, dtype=np.int64); rt = np.asarray(rt, dtype=np.float32)
            ctx_items, ctx_rt = it[~tm], rt[~tm]
            if len(ctx_items) < min_pool:
                continue
            self.samples.append((ctx_items, ctx_rt, it[tm], rt[tm]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ctx_items, ctx_rt, tgt_items, tgt_true = self.samples[i]
        b_u = float((ctx_rt - self.mu - self.b_i[ctx_items]).mean())

        items = np.concatenate([ctx_items, tgt_items])
        L, T = len(ctx_items), len(tgt_items)
        levels = np.zeros(len(items), dtype=np.int64)
        bias = np.zeros(len(items), dtype=np.float32)
        levels[:L] = _rating_level(ctx_rt)                 # cibles rating-free (level 0)
        bias[:L] = ctx_rt - float(ctx_rt.mean())

        target_mask = np.zeros(len(items), dtype=bool)
        target_mask[L:] = True
        baseline = self.mu + self.b_i[tgt_items] + b_u     # μ+b_i+b_u par cible
        target_baseline = np.zeros(len(items), dtype=np.float32)
        target_true = np.zeros(len(items), dtype=np.float32)
        target_baseline[L:] = baseline
        target_true[L:] = tgt_true

        return {
            "items": torch.from_numpy(items),
            "levels": torch.from_numpy(levels),
            "bias": torch.from_numpy(bias),
            "target_mask": torch.from_numpy(target_mask),
            "target_baseline": torch.from_numpy(target_baseline),
            "target_true": torch.from_numpy(target_true),
        }


# --------------------------------------------------------------------------- #
# Collate : padding au max du lot + key_padding_mask
# --------------------------------------------------------------------------- #
def collate(batch, pad_item: int = 0):
    """Pad la dimension film. Renvoie un dict de tenseurs (B, Lmax) + key_padding_mask.

    `key_padding_mask` : True = padding (ignoré par l'attention), convention PyTorch.
    Les clés float optionnelles (target_baseline/target_true) sont incluses si présentes.
    """
    keys = batch[0].keys()
    B = len(batch)
    Lmax = max(b["items"].shape[0] for b in batch)
    out = {}
    dtypes = {"items": torch.long, "levels": torch.long, "bias": torch.float32,
              "target_mask": torch.bool, "target_resid": torch.float32,
              "target_baseline": torch.float32, "target_true": torch.float32}
    for k in keys:
        buf = torch.zeros((B, Lmax), dtype=dtypes[k])
        if k == "items":
            buf.fill_(pad_item)
        for i, b in enumerate(batch):
            n = b[k].shape[0]
            buf[i, :n] = b[k]
        out[k] = buf
    key_padding_mask = torch.ones((B, Lmax), dtype=torch.bool)
    for i, b in enumerate(batch):
        key_padding_mask[i, :b["items"].shape[0]] = False
    out["key_padding_mask"] = key_padding_mask
    return out
