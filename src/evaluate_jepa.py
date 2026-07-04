"""Évaluation POC du JEPA de trajectoires.

Deux familles de mesures :
1. RETRIEVAL — la représentation prédite ẑ retrouve-t-elle le futur ?
   - niveau films : banque d'embeddings par film (chunk singleton via encodeur cible) ;
     Recall@K et rang médian des films du vrai prochain chunk.
   - niveau chunks : plus proche voisin de ẑ parmi une banque de chunks réels.
   Baselines : popularité (films les + vus) et répétition (contenu du dernier chunk vu).
2. PROBE — l'espace latent code-t-il le contenu ?
   - régression linéaire hₜ -> genome moyen du prochain chunk (R²)
   - classification linéaire hₜ -> genres du prochain chunk (AUC macro)
   - projection UMAP des représentations de chunk, colorée par genre dominant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data_prep import DATA, load_all
from .data_jepa import JepaEvalDataset, collate_eval, to_chunks, CHUNK_SIZE


# --------------------------------------------------------------------------- #
# Encodages de base
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_item_bank(model, device, batch: int = 2048) -> torch.Tensor:
    """Chaque film encodé comme un chunk singleton (via l'encodeur CIBLE) -> (n_items+1, d).

    On répète le film K fois pour former un 'chunk' homogène ; l'encodeur cible en
    donne un point dans le MÊME espace que les cibles z. Ligne 0 (padding) incluse.
    """
    model.eval()
    n = model.target_chunk.tokenizer.id_emb.num_embeddings
    K = CHUNK_SIZE
    banks = []
    for start in range(0, n, batch):
        ids = torch.arange(start, min(start + batch, n), device=device)
        chunk = ids[:, None].repeat(1, K)                 # (b, K) : film répété K fois
        banks.append(model.target_chunk(chunk))
    return torch.cat(banks, dim=0)                        # (n, d)


@torch.no_grad()
def encode_eval(model, sequences, device, split: str = "test", batch: int = 128):
    """Encode les contextes d'éval -> (ẑ prédit, h contexte, chunk cible items).

    ẑ = prédiction du prochain chunk à la DERNIÈRE position de contexte de chaque user.
    """
    model.eval()
    ds = JepaEvalDataset(sequences, split=split, K=CHUNK_SIZE, min_chunks=3)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_eval)
    Z, H, T = [], [], []
    for chunks, mask, target in loader:
        chunks, mask = chunks.to(device), mask.to(device)
        c = model.online_chunk(chunks.reshape(-1, CHUNK_SIZE)).reshape(chunks.shape[0], chunks.shape[1], -1)
        h = model.temporal(c, ~mask)
        zhat = model.predictor(h)
        last = mask.sum(dim=1) - 1                         # index du dernier chunk réel
        idx = torch.arange(chunks.shape[0], device=device)
        Z.append(zhat[idx, last].cpu())
        H.append(h[idx, last].cpu())
        T.append(target)
    return torch.cat(Z), torch.cat(H), torch.cat(T)        # (U,d), (U,d), (U,K)


# --------------------------------------------------------------------------- #
# 1. Retrieval niveau films
# --------------------------------------------------------------------------- #
def _popularity(sequences, n_items: int) -> np.ndarray:
    counts = np.zeros(n_items + 1, dtype=np.int64)
    for items in sequences["items"].values:
        np.add.at(counts, np.asarray(items), 1)
    counts[0] = -1                                         # padding jamais recommandé
    return counts


def retrieval_films(zhat, item_bank, target_chunks, sequences, maps,
                    ks=(10, 20, 50, 100), device="cpu"):
    """Recall@K et rang médian des films du vrai prochain chunk, vs baselines."""
    bank = F.normalize(item_bank.to(device), dim=-1)       # (n+1, d)
    q = F.normalize(zhat.to(device), dim=-1)               # (U, d)
    scores = q @ bank.T                                    # (U, n+1)
    scores[:, 0] = -1e9                                    # exclut le padding
    U = q.shape[0]

    # baseline popularité (même ranking pour tous)
    pop = _popularity(sequences, maps.n_items)
    pop_rank = np.argsort(-pop)                            # films triés par popularité

    out = {}
    ranks_model = []
    hit = {k: 0 for k in ks}
    hit_pop = {k: 0 for k in ks}
    topk_max = max(ks)
    order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()   # (U, n+1)
    pop_top = set(pop_rank[:topk_max].tolist())

    for u in range(U):
        truth = set(int(i) for i in target_chunks[u].tolist() if int(i) != 0)
        if not truth:
            continue
        ranked = order[u]
        pos = {int(film): r for r, film in enumerate(ranked[:topk_max])}
        # rang médian du meilleur film vrai (modèle)
        best = min((pos.get(f, topk_max) for f in truth), default=topk_max)
        ranks_model.append(best)
        for k in ks:
            topk = set(int(x) for x in ranked[:k])
            if truth & topk:
                hit[k] += 1
            if truth & set(pop_rank[:k].tolist()):
                hit_pop[k] += 1

    out["recall_model"] = {k: hit[k] / U for k in ks}
    out["recall_pop"] = {k: hit_pop[k] / U for k in ks}
    out["median_rank_model"] = float(np.median(ranks_model))
    return out


# --------------------------------------------------------------------------- #
# 2. Retrieval niveau chunks (+ baseline répétition)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def retrieval_chunks(model, zhat, sequences, maps, device, ks=(1, 5, 10, 20)):
    """Plus proche voisin de ẑ parmi une banque de chunks réels (test).

    On construit la banque des VRAIS chunks cibles (un par user) ; la bonne réponse
    pour le user u est son propre chunk. Baseline répétition : le dernier chunk du
    contexte comme prédiction.
    """
    ds = JepaEvalDataset(sequences, split="test", K=CHUNK_SIZE, min_chunks=3)
    # banque = encodage cible des chunks vrais ; requêtes = ẑ
    tgt_chunks = torch.stack([torch.as_tensor(ds.target[i]) for i in range(len(ds))])  # (U,K)
    bank = F.normalize(model.target_chunk(tgt_chunks.to(device)).cpu(), dim=-1)         # (U,d)
    q = F.normalize(zhat, dim=-1)
    sims = q @ bank.T                                       # (U, U)
    U = q.shape[0]
    ranks = (sims.argsort(dim=1, descending=True) == torch.arange(U)[:, None]).float().argmax(1)
    recall = {k: float((ranks < k).float().mean()) for k in ks}

    # baseline répétition : dernier chunk de contexte -> encodé cible -> NN dans la banque
    last_ctx = torch.stack([torch.as_tensor(ds.context[i][-1]) for i in range(len(ds))])
    qrep = F.normalize(model.target_chunk(last_ctx.to(device)).cpu(), dim=-1)
    sims_r = qrep @ bank.T
    ranks_r = (sims_r.argsort(dim=1, descending=True) == torch.arange(U)[:, None]).float().argmax(1)
    recall_rep = {k: float((ranks_r < k).float().mean()) for k in ks}
    return {"recall_model": recall, "recall_repeat": recall_rep, "n": U}


# --------------------------------------------------------------------------- #
# 3. Probes de contenu
# --------------------------------------------------------------------------- #
def _chunk_genome(target_chunks, genome):
    """Genome moyen des films de chaque chunk cible -> (U, n_tags)."""
    out = np.zeros((len(target_chunks), genome.shape[1]), dtype=np.float32)
    for i, ch in enumerate(target_chunks):
        ids = [int(x) for x in ch.tolist() if int(x) != 0]
        if ids:
            out[i] = genome[ids].mean(axis=0)
    return out


def probe_genome(H, target_chunks, genome, n_train=0.8):
    """Régression linéaire h -> genome moyen du prochain chunk. R² (test)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X = H.numpy()
    Y = _chunk_genome(target_chunks, genome)
    cut = int(len(X) * n_train)
    reg = Ridge(alpha=10.0).fit(X[:cut], Y[:cut])
    pred = reg.predict(X[cut:])
    return float(r2_score(Y[cut:], pred, multioutput="variance_weighted"))


def probe_genres(H, target_chunks, maps, n_train=0.8, top_genres=12):
    """Classification linéaire h -> genres du prochain chunk. AUC macro (test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    movie = pd.read_csv(DATA / "movie.csv")
    genre_by_movie = dict(zip(movie["movieId"], movie["genres"]))
    # genres du chunk (multi-label) via idx interne -> movieId -> genres
    all_genres = {}
    labels = []
    for ch in target_chunks:
        gs = set()
        for x in ch.tolist():
            x = int(x)
            if x == 0:
                continue
            g = genre_by_movie.get(maps.idx2movie[x], "")
            gs.update(g.split("|"))
        labels.append(gs)
    # garder les genres les plus fréquents
    from collections import Counter
    freq = Counter(g for s in labels for g in s if g and g != "(no genres listed)")
    keep = [g for g, _ in freq.most_common(top_genres)]
    Y = np.array([[1 if g in s else 0 for g in keep] for s in labels])

    X = H.numpy()
    cut = int(len(X) * n_train)
    aucs = []
    for j in range(Y.shape[1]):
        if Y[:cut, j].sum() == 0 or Y[cut:, j].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[:cut], Y[:cut, j])
        p = clf.predict_proba(X[cut:])[:, 1]
        aucs.append(roc_auc_score(Y[cut:, j], p))
    return float(np.mean(aucs)), keep


# --------------------------------------------------------------------------- #
# 4. Projection UMAP des chunks
# --------------------------------------------------------------------------- #
@torch.no_grad()
def umap_chunks(model, sequences, maps, device, n_chunks=4000, seed=0):
    """Encode des chunks réels et renvoie une projection 2D UMAP + genre dominant."""
    import umap

    movie = pd.read_csv(DATA / "movie.csv")
    genre_by_movie = dict(zip(movie["movieId"], movie["genres"]))

    rng = np.random.default_rng(seed)
    chunks, genres = [], []
    for items in sequences["items"].sample(min(2000, len(sequences)), random_state=seed).values:
        ch = to_chunks(items, CHUNK_SIZE)
        for c in ch:
            chunks.append(c)
            first = genre_by_movie.get(maps.idx2movie[int(c[0])], "").split("|")[0]
            genres.append(first)
        if len(chunks) >= n_chunks:
            break
    chunks = torch.as_tensor(np.stack(chunks[:n_chunks]))
    emb = model.target_chunk(chunks.to(device)).cpu().numpy()
    proj = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(emb)
    return proj, genres[:n_chunks]
