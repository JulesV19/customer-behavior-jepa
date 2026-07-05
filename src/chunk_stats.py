"""Analyse MODEL-FREE du signal de base dans les trajectoires de chunks.

Objectif : AVANT tout modèle, mesurer si le contenu des chunks d'un même user
dérive de façon structurée dans le temps. Autrement dit : deux chunks voisins
(t, t+1) sont-ils plus proches que deux chunks au hasard ? Si oui, il y a
quelque chose à prédire et le JEPA a une raison d'exister ; sinon, le signal
est plat et aucun modèle ne pourra prédire le prochain chunk.

Tout est calculé à partir du CONTENU brut (vecteur genome, genre, note,
popularité), sans jamais toucher au checkpoint entraîné. Un chunk = K films
consécutifs (cf. data_jepa.to_chunks) ; le chunk t couvre items[t*K:(t+1)*K].

Note cosine : les vecteurs genome sont des relevances dans [0,1] (tout positif),
donc le cosine BRUT entre deux chunks est mécaniquement élevé (l'espace est
dans l'orthant positif). On CENTRE par le genome moyen global avant de mesurer
le cosine (`center=True`), pour ne garder que la structure et pas l'offset commun
— même logique que le centering de la perte JEPA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import DATA, IdMaps
from .data_jepa import to_chunks, CHUNK_SIZE


# --------------------------------------------------------------------------- #
# Briques : popularité, genres, genome moyen par chunk
# --------------------------------------------------------------------------- #
def item_popularity(sequences: pd.DataFrame, n_items: int) -> np.ndarray:
    """Nombre de vues par item (index interne) sur tout le corpus. Ligne 0 = padding."""
    counts = np.zeros(n_items + 1, dtype=np.int64)
    for items in sequences["items"].values:
        np.add.at(counts, np.asarray(items, dtype=np.int64), 1)
    counts[0] = 0
    return counts


def build_genre_by_idx(maps: IdMaps) -> dict[int, str]:
    """index interne -> premier genre du film (proxy de genre dominant)."""
    movie = pd.read_csv(DATA / "movie.csv", usecols=["movieId", "genres"])
    genre_by_movie = dict(zip(movie["movieId"], movie["genres"]))
    out = {}
    for idx, mid in maps.idx2movie.items():
        g = genre_by_movie.get(mid, "")
        first = g.split("|")[0] if g else ""
        out[idx] = first if first and first != "(no genres listed)" else "Unknown"
    return out


def chunk_genome(items, genome: np.ndarray, K: int = CHUNK_SIZE) -> np.ndarray:
    """Genome moyen de chaque chunk d'un user -> (M, n_tags)."""
    ch = to_chunks(items, K)                                   # (M, K)
    if len(ch) == 0:
        return np.zeros((0, genome.shape[1]), dtype=np.float32)
    return genome[ch].mean(axis=1)                            # (M, n_tags)


def _pairwise_diversity(vecs: np.ndarray) -> float:
    """Diversité intra-chunk : 1 - cosine moyen entre les K films du chunk.

    0 = films identiques en contenu ; ~1 = films très hétérogènes.
    """
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    u = vecs / norm
    sims = u @ u.T
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sims[iu].mean())


def user_chunk_scalars(items, ratings, genome, logpop, genre_by_idx,
                       K: int = CHUNK_SIZE) -> dict:
    """Stats scalaires par chunk pour un user : note, popularité, diversité, genre.

    Renvoie des tableaux de longueur M (nb de chunks pleins) :
      rating (M,), logpop (M,), diversity (M,), genre (list[str] len M).
    """
    ch = to_chunks(items, K)                                  # (M, K)
    M = len(ch)
    ratings = np.asarray(ratings, dtype=np.float32)
    rating = np.array([ratings[t * K:(t + 1) * K].mean() for t in range(M)], dtype=np.float32)
    lp = np.array([logpop[ch[t]].mean() for t in range(M)], dtype=np.float32)
    div = np.array([_pairwise_diversity(genome[ch[t]]) for t in range(M)], dtype=np.float32)
    genres = []
    for t in range(M):
        gs = [genre_by_idx.get(int(i), "Unknown") for i in ch[t]]
        # genre dominant du chunk = le plus fréquent parmi ses K films
        genres.append(max(set(gs), key=gs.count))
    return {"rating": rating, "logpop": lp, "diversity": div, "genre": genres, "M": M}


# --------------------------------------------------------------------------- #
# Centre global (pour un cosine discriminant)
# --------------------------------------------------------------------------- #
def global_mean_chunk_genome(sequences, genome, K: int = CHUNK_SIZE,
                             n_chunks: int = 50000, seed: int = 0) -> np.ndarray:
    """Genome moyen d'un échantillon de chunks -> vecteur (n_tags,) à retrancher."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    acc = np.zeros(genome.shape[1], dtype=np.float64)
    n = 0
    for i in order:
        g = chunk_genome(sequences["items"].values[i], genome, K)
        if len(g):
            acc += g.sum(axis=0)
            n += len(g)
        if n >= n_chunks:
            break
    return (acc / max(1, n)).astype(np.float32)


def _normed(g: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    """Centre (optionnel) puis L2-normalise les lignes -> cosine = produit scalaire."""
    if center is not None:
        g = g - center[None, :]
    norm = np.linalg.norm(g, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return g / norm


# --------------------------------------------------------------------------- #
# 1. Autocorrélation : similarité chunk t vs chunk t+lag
# --------------------------------------------------------------------------- #
def autocorr_by_lag(sequences, genome, K: int = CHUNK_SIZE, max_lag: int = 10,
                    n_users: int = 5000, center: np.ndarray | None = None,
                    seed: int = 0) -> dict:
    """Similarité cosine moyenne entre chunks distants de `lag`, vs deux baselines.

    - real        : ordre temporel réel, moyenne de cos(g_t, g_{t+lag}) sur tous les
                    users et positions. Si ça DÉCROÎT avec le lag -> mémoire temporelle.
    - shuffled    : mêmes chunks mais ordre MÉLANGÉ dans chaque user. Détruit le temps,
                    garde le goût du user -> plancher "signal user, sans temps".
    - cross_user  : cosine entre chunks de users DIFFÉRENTS au hasard -> plancher global.

    Si real(lag=1) > shuffled ~ constante > cross_user : il y a bien un signal
    temporel EN PLUS du signal d'identité du user.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))[:n_users]

    lags = np.arange(1, max_lag + 1)
    real_sum = np.zeros(max_lag + 1); real_cnt = np.zeros(max_lag + 1)
    shuf_sum = np.zeros(max_lag + 1); shuf_cnt = np.zeros(max_lag + 1)
    pool = []                                                 # 1 chunk/user pour le cross-user

    for i in idx:
        g = chunk_genome(sequences["items"].values[i], genome, K)
        if len(g) < 2:
            continue
        u = _normed(g, center)                                # (M, d)
        M = u.shape[0]
        sims = u @ u.T                                        # (M, M)
        for lag in lags:
            if lag < M:
                d = np.diagonal(sims, offset=lag)
                real_sum[lag] += d.sum(); real_cnt[lag] += d.size
        # baseline shuffle intra-user
        us = u[rng.permutation(M)]
        sims_s = us @ us.T
        for lag in lags:
            if lag < M:
                d = np.diagonal(sims_s, offset=lag)
                shuf_sum[lag] += d.sum(); shuf_cnt[lag] += d.size
        pool.append(u[rng.integers(M)])

    real = real_sum[1:] / np.maximum(1, real_cnt[1:])
    shuf = shuf_sum[1:] / np.maximum(1, shuf_cnt[1:])
    # cross-user : paires aléatoires de chunks de users différents
    P = np.stack(pool) if pool else np.zeros((0, genome.shape[1]))
    if len(P) > 1:
        perm = rng.permutation(len(P))
        cross = float((P * P[perm]).sum(axis=1).mean())
    else:
        cross = float("nan")
    return {"lags": lags, "real": real, "shuffled": shuf, "cross_user": cross,
            "n_users": int(len(pool))}


# --------------------------------------------------------------------------- #
# 2. Heatmap de dérive : similarité moyenne chunk_i x chunk_j
# --------------------------------------------------------------------------- #
def similarity_heatmap(sequences, genome, K: int = CHUNK_SIZE, P: int = 12,
                       n_users: int = 5000, center: np.ndarray | None = None,
                       seed: int = 0) -> np.ndarray:
    """Matrice (P, P) : cosine moyen entre le i-ème et le j-ème chunk des users.

    Ne garde que les users ayant >= P chunks, aligne sur les P PREMIERS chunks.
    Une diagonale marquée = mémoire courte (voisins proches) ; des blocs = phases
    de goût stables ; une matrice uniforme = pas de structure temporelle.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))[:n_users]
    acc = np.zeros((P, P), dtype=np.float64)
    n = 0
    for i in idx:
        g = chunk_genome(sequences["items"].values[i], genome, K)
        if len(g) < P:
            continue
        u = _normed(g[:P], center)                            # (P, d)
        acc += u @ u.T
        n += 1
    return (acc / max(1, n)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Trajectoires d'exemples : stats par chunk pour quelques users
# --------------------------------------------------------------------------- #
def example_trajectories(sequences, genome, logpop, genre_by_idx,
                         user_rows: list[int], K: int = CHUNK_SIZE,
                         center: np.ndarray | None = None) -> list[dict]:
    """Pour chaque user (ligne du df) : stats scalaires par chunk + similarité au
    chunk précédent, prêtes à tracer (X = index de chunk)."""
    out = []
    for r in user_rows:
        items = sequences["items"].values[r]
        ratings = sequences["ratings"].values[r]
        sc = user_chunk_scalars(items, ratings, genome, logpop, genre_by_idx, K)
        g = chunk_genome(items, genome, K)
        u = _normed(g, center)
        prev_sim = np.full(sc["M"], np.nan, dtype=np.float32)
        if sc["M"] >= 2:
            prev_sim[1:] = (u[1:] * u[:-1]).sum(axis=1)       # cos(chunk_t, chunk_{t-1})
        out.append({
            "user_row": int(r),
            "userId": int(sequences["userId"].values[r]),
            "M": sc["M"],
            "rating": sc["rating"],
            "logpop": sc["logpop"],
            "diversity": sc["diversity"],
            "genre": sc["genre"],
            "prev_sim": prev_sim,
        })
    return out
