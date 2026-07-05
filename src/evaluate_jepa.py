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
def representation_diagnostics(model, sequences, device, n_chunks: int = 5000, seed: int = 0):
    """Diagnostic d'anisotropie : à quel point les représentations se ressemblent-elles ?

    Si la similarité cosine MOYENNE entre deux chunks au hasard est déjà élevée (~0.9),
    alors un cosine de prédiction de 0.97 est TRIVIAL (l'espace est étroit) -> raccourci.
    Si elle est basse (~0.2), un cosine élevé de prédiction est un vrai signal.
    On mesure aussi l'écart entre 'similarité au bon chunk' et 'similarité à un chunk au hasard'.
    """
    from .data_jepa import to_chunks as _tc
    rng = np.random.default_rng(seed)
    chunks = []
    for items in sequences.sample(min(3000, len(sequences)), random_state=seed)["items"].values:
        for c in _tc(items, CHUNK_SIZE):
            chunks.append(c)
        if len(chunks) >= n_chunks:
            break
    chunks = torch.as_tensor(np.stack(chunks[:n_chunks]))
    z = F.normalize(model.target_chunk(chunks.to(device)).cpu(), dim=-1)   # cibles normalisées
    # similarité cosine moyenne entre paires aléatoires
    perm = torch.randperm(z.shape[0])
    rand_sim = (z * z[perm]).sum(-1).mean().item()
    # anisotropie globale : norme du vecteur moyen (1 = tout aligné, 0 = isotrope)
    mean_dir_norm = z.mean(0).norm().item()
    return {"mean_pairwise_cos_targets": rand_sim, "anisotropy_mean_dir_norm": mean_dir_norm,
            "n_chunks": z.shape[0]}


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
    for chunks, mask, target, levels, bias in loader:
        chunks, mask = chunks.to(device), mask.to(device)
        lv = levels.to(device).reshape(-1, CHUNK_SIZE) if model.use_ratings else None
        bs = bias.to(device).reshape(-1, CHUNK_SIZE) if model.use_ratings else None
        c = model.online_chunk(chunks.reshape(-1, CHUNK_SIZE), lv, bs) \
            .reshape(chunks.shape[0], chunks.shape[1], -1)
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
# 1bis. Baseline kNN-CONTENU (model-free) : "encore du même contenu"
# --------------------------------------------------------------------------- #
def _chunk_mean_genome(chunk_items, genome: np.ndarray) -> np.ndarray:
    """Genome moyen des films d'un chunk (ignore le padding). -> (n_tags,)."""
    ids = [int(x) for x in np.asarray(chunk_items).tolist() if int(x) != 0]
    if not ids:
        return np.zeros(genome.shape[1], dtype=np.float32)
    return genome[ids].mean(axis=0)


def content_knn_retrieval(sequences_eval, genome, maps, split: str = "test",
                          ks_film=(10, 20, 50, 100), ks_chunk=(1, 5, 10, 20),
                          device="cpu"):
    """Baseline SANS MODÈLE : la requête est le genome moyen du DERNIER chunk de contexte.

    C'est l'heuristique "persistance de contenu" (le signal court-portée du notebook 04),
    sans rien apprendre. On classe :
      - films  : par cosine(genome dernier chunk, genome du film) -> Recall@K, rang médian ;
      - chunks : plus proche voisin parmi les vrais chunks cibles (genome moyen).
    Même protocole/même échantillon que le JEPA -> comparaison apples-to-apples.
    Si le JEPA bat cette baseline, il apporte + que "encore du même contenu".
    """
    ds = JepaEvalDataset(sequences_eval, split=split, K=CHUNK_SIZE, min_chunks=3)
    U = len(ds)
    g = torch.as_tensor(genome)                               # (n+1, n_tags)

    # requête = genome moyen du dernier chunk de contexte de chaque user
    q = torch.stack([torch.as_tensor(_chunk_mean_genome(ds.context[i][-1], genome))
                     for i in range(U)])                      # (U, n_tags)
    qn = F.normalize(q.to(device), dim=-1)

    # ---- films ----
    bank = F.normalize(g.to(device), dim=-1)                  # (n+1, n_tags)
    scores = qn @ bank.T                                      # (U, n+1)
    scores[:, 0] = -1e9                                       # exclut padding
    order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
    topk_max = max(ks_film)
    hit = {k: 0 for k in ks_film}
    ranks = []
    for u in range(U):
        truth = set(int(i) for i in np.asarray(ds.target[u]).tolist() if int(i) != 0)
        if not truth:
            continue
        ranked = order[u]
        pos = {int(f): r for r, f in enumerate(ranked[:topk_max])}
        ranks.append(min((pos.get(f, topk_max) for f in truth), default=topk_max))
        for k in ks_film:
            if truth & set(int(x) for x in ranked[:k]):
                hit[k] += 1
    recall_film = {k: hit[k] / U for k in ks_film}
    median_rank_film = float(np.median(ranks))

    # ---- chunks ----
    tgt = torch.stack([torch.as_tensor(_chunk_mean_genome(ds.target[i], genome))
                       for i in range(U)])                    # (U, n_tags)
    tn = F.normalize(tgt.to(device), dim=-1)
    sims = qn @ tn.T                                          # (U, U)
    rank_self = (sims.argsort(dim=1, descending=True) == torch.arange(U, device=device)[:, None]) \
        .float().argmax(1)
    recall_chunk = {k: float((rank_self < k).float().mean()) for k in ks_chunk}

    return {"recall_film": recall_film, "median_rank_film": median_rank_film,
            "recall_chunk": recall_chunk, "n": U}


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


@torch.no_grad()
def pca_chunks(model, sequences, maps, device, n_chunks=4000, seed=0, n_components=3):
    """Contrôle LINÉAIRE de l'UMAP : même échantillon de chunks, projection PCA.

    Déterministe et sans hyperparamètre. Si les genres se séparent DÉJÀ en PCA, la
    structure est linéairement lisible (signal fort) ; sinon, la non-linéarité de
    l'UMAP était nécessaire. On renvoie aussi la variance expliquée par chaque axe.
    """
    from sklearn.decomposition import PCA

    movie = pd.read_csv(DATA / "movie.csv")
    genre_by_movie = dict(zip(movie["movieId"], movie["genres"]))

    chunks, genres = [], []
    # MÊME tirage que umap_chunks (même seed) -> exactement les mêmes points encodés
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
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(emb)
    return proj, genres[:n_chunks], pca.explained_variance_ratio_


# --------------------------------------------------------------------------- #
# 5. Suite complète de métriques (pour comparer plusieurs modèles)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_model(model, sequences, maps, genome, device, eval_users: int = 8000,
                   seed: int = 0, ks_film=(10, 20, 50, 100), ks_chunk=(1, 5, 10, 20)) -> dict:
    """Calcule toute la suite d'éval pour UN modèle -> dict de métriques scalaires.

    Réutilise les fonctions existantes ; conçu pour être appelé une fois par modèle
    afin de construire un tableau comparatif. Les baselines model-free (popularité,
    répétition, kNN-contenu) sont incluses ici mais IDENTIQUES entre modèles.
    """
    eval_sub = sequences.sample(eval_users, random_state=seed).reset_index(drop=True)
    diag = representation_diagnostics(model, sequences, device, seed=seed)
    item_bank = build_item_bank(model, device)
    zhat, H, T = encode_eval(model, eval_sub, device, split="test")
    rf = retrieval_films(zhat, item_bank, T, sequences, maps, ks=ks_film, device=device)
    rc = retrieval_chunks(model, zhat, eval_sub, maps, device, ks=ks_chunk)
    r2 = probe_genome(H, T, genome)
    auc, _ = probe_genres(H, T, maps)
    return {
        "rand_cos": diag["mean_pairwise_cos_targets"],
        "anisotropy": diag["anisotropy_mean_dir_norm"],
        "recall_film": rf["recall_model"], "recall_pop": rf["recall_pop"],
        "median_rank_film": rf["median_rank_model"],
        "recall_chunk": rc["recall_model"], "recall_repeat": rc["recall_repeat"],
        "probe_genome_r2": r2, "probe_genres_auc": auc, "n_eval": len(eval_sub),
    }
