"""Analyse des habitudes temporelles de NOTATION des users MovieLens.

Motivation : le chunk de taille K=5 (cf. `data_jepa.to_chunks`) est un choix
ARBITRAIRE. Les timestamps MovieLens sont des dates de notation en RAFALES
(pas de dates de visionnage). Ce module mesure la structure réelle de ces
rafales pour répondre à trois questions de design :

  1. Y a-t-il une taille de rafale NATURELLE (et vaut-elle ~5) ?
  2. Existe-t-il une "vallée" nette dans les écarts qui sépare le "dans la même
     session" du "revenu plus tard" -> un seuil de session non-arbitraire ?
  3. Le K=5 fixe RESPECTE-t-il les frontières de session, ou coupe-t-il au
     milieu d'une rafale (auquel cas l'hypothèse "chunk = un ensemble d'un même
     moment, sans ordre interne" est violée) ?

Tout est calculé à partir de `sequences['timestamps']` (unix secondes, déjà
trié par user puis timestamp, tronqué aux 500 derniers items). On ne retouche
PAS au `rating.csv` de 690 Mo.

Vocabulaire :
  - gap     : écart en secondes entre deux notations consécutives d'un user.
  - session : suite maximale de notations dont les gaps internes sont < tau
              (rafale). Une frontière de session = un gap >= tau.
  - chunk   : K items consécutifs (exactement comme le modèle, `to_chunks`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_jepa import CHUNK_SIZE

# Seuils de session usuels, en secondes, avec libellés lisibles.
TAUS: dict[str, int] = {
    "1 min": 60,
    "10 min": 600,
    "1 h": 3600,
    "1 jour": 86400,
    "1 semaine": 604800,
}


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _sample_ts(sequences: pd.DataFrame, sample: int | None, seed: int) -> np.ndarray:
    """Renvoie l'array (object) des listes de timestamps, éventuellement échantillonné."""
    ts_col = sequences["timestamps"].values
    if sample is not None and sample < len(ts_col):
        rng = np.random.default_rng(seed)
        ts_col = ts_col[rng.permutation(len(ts_col))[:sample]]
    return ts_col


def human_seconds(s: float) -> str:
    """Formate une durée en secondes -> libellé court lisible ('3 j', '12 min')."""
    for unit, sec in (("an", 31_557_600), ("j", 86_400), ("h", 3_600), ("min", 60)):
        if s >= sec:
            return f"{s / sec:.0f} {unit}"
    return f"{s:.0f} s"


# --------------------------------------------------------------------------- #
# 1. Distribution brute des écarts entre notations
# --------------------------------------------------------------------------- #
def all_gaps(sequences: pd.DataFrame, sample: int | None = None,
             seed: int = 0) -> np.ndarray:
    """Tous les gaps (secondes) entre notations consécutives, mis en commun.

    sample = nb de users tirés (None = tous ; le calcul est bon marché). 1-D array.
    """
    out = []
    for ts in _sample_ts(sequences, sample, seed):
        a = np.asarray(ts, dtype=np.int64)
        if a.size >= 2:
            out.append(np.diff(a))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 2. Segmentation en sessions / rafales
# --------------------------------------------------------------------------- #
def _session_sizes(ts, tau: int) -> np.ndarray:
    """Tailles des sessions d'un user : on coupe dès qu'un gap >= tau."""
    a = np.asarray(ts, dtype=np.int64)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(a) >= tau) + 1
    edges = np.concatenate(([0], boundaries, [a.size]))
    return np.diff(edges)


def burst_sizes(sequences: pd.DataFrame, tau: int, sample: int | None = 40000,
                seed: int = 0) -> np.ndarray:
    """Distribution (mise en commun) des tailles de session pour un seuil `tau`."""
    sizes = [_session_sizes(ts, tau) for ts in _sample_ts(sequences, sample, seed)]
    return np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int64)


def burst_size_summary(sequences: pd.DataFrame, taus: dict[str, int] = TAUS,
                       sample: int | None = 40000, seed: int = 0,
                       K: int = CHUNK_SIZE) -> pd.DataFrame:
    """Pour chaque seuil `tau` : stats de taille de session + position de K.

    `%_notes_en_sessions>=K` = part des NOTATIONS (pondérée par taille) qui vivent
    dans une session d'au moins K items : au-dessous, K=5 est trop grand pour la
    majorité des rafales ; au-dessus, les rafales dépassent K et K=5 les fragmente.
    """
    rows = []
    for label, tau in taus.items():
        s = burst_sizes(sequences, tau, sample=sample, seed=seed)
        n = s.size
        tot = int(s.sum())
        rows.append({
            "seuil": label,
            "tau_s": tau,
            "n_sessions": n,
            "médiane": float(np.median(s)) if n else np.nan,
            "moyenne": float(s.mean()) if n else np.nan,
            "p90": float(np.percentile(s, 90)) if n else np.nan,
            "max": int(s.max()) if n else 0,
            "%_taille_1": float((s == 1).mean()) if n else np.nan,
            f"%_==K({K})": float((s == K).mean()) if n else np.nan,
            "%_<K": float((s < K).mean()) if n else np.nan,
            "%_>K": float((s > K).mean()) if n else np.nan,
            "%_notes_en_sessions>=K": float(s[s >= K].sum() / tot) if tot else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Notations simultanées (même seconde) : l'ordre intra-rafale est-il réel ?
# --------------------------------------------------------------------------- #
def same_timestamp_stats(sequences: pd.DataFrame, sample: int | None = 40000,
                         seed: int = 0) -> dict:
    """Part des gaps nuls et taille des paquets de notes à la MÊME seconde.

    Beaucoup de gaps == 0 => l'ordre intra-rafale est en partie arbitraire
    (notées d'un bloc) : justifie l'encodeur de chunk SANS position.
    """
    gaps = all_gaps(sequences, sample=sample, seed=seed)
    sizes = burst_sizes(sequences, tau=1, sample=sample, seed=seed)   # runs même-seconde
    return {
        "frac_gaps_zero": float((gaps == 0).mean()) if gaps.size else np.nan,
        "frac_gaps_<=60s": float((gaps <= 60).mean()) if gaps.size else np.nan,
        "frac_gaps_<=1h": float((gaps <= 3600).mean()) if gaps.size else np.nan,
        "frac_gaps_<=1j": float((gaps <= 86400).mean()) if gaps.size else np.nan,
        "médiane_paquet_même_seconde": float(np.median(sizes)) if sizes.size else np.nan,
        "p90_paquet_même_seconde": float(np.percentile(sizes, 90)) if sizes.size else np.nan,
        "max_paquet_même_seconde": int(sizes.max()) if sizes.size else 0,
    }


# --------------------------------------------------------------------------- #
# 4. Ce que le K fixe fait aux rafales réelles
# --------------------------------------------------------------------------- #
def _chunk_internal_maxgap(ts, K: int) -> np.ndarray:
    """Pour chaque chunk plein (K items consécutifs) : plus grand gap INTERNE (s).

    Aligné exactement sur `to_chunks` : chunks non chevauchants, reste ignoré.
    """
    a = np.asarray(ts, dtype=np.int64)
    M = a.size // K
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    a = a[:M * K].reshape(M, K)
    return np.diff(a, axis=1).max(axis=1)          # (M,) : max des K-1 gaps internes


def chunk_purity_summary(sequences: pd.DataFrame, Ks=(3, 4, 5, 6, 8, 10),
                         tau: int = 3600, sample: int | None = 40000,
                         seed: int = 0) -> pd.DataFrame:
    """Pour chaque K : un chunk de K items reste-t-il DANS une seule session ?

    Un chunk est 'propre' si son plus grand gap interne < tau (les K items sont
    de la même rafale). Sinon il CHEVAUCHE une frontière de session : il mélange
    deux moments distincts, ce que l'archi (chunk = ensemble non ordonné) suppose
    justement ne PAS se produire. Plus K grand, plus le risque d'impureté monte :
    le tableau montre le compromis.
    """
    ts_col = _sample_ts(sequences, sample, seed)
    rows = []
    for K in Ks:
        mg = [_chunk_internal_maxgap(ts, K) for ts in ts_col]
        mg = np.concatenate(mg) if mg else np.zeros(0, dtype=np.int64)
        n = mg.size
        rows.append({
            "K": K,
            "n_chunks": n,
            "%_chunks_propres(<tau)": float((mg < tau).mean()) if n else np.nan,
            "%_chunks_impurs(>=tau)": float((mg >= tau).mean()) if n else np.nan,
            "médiane_maxgap_h": float(np.median(mg) / 3600) if n else np.nan,
            "p90_maxgap_j": float(np.percentile(mg, 90) / 86400) if n else np.nan,
        })
    return pd.DataFrame(rows)
