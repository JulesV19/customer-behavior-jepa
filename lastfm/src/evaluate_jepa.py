"""Évaluation du JEPA d'écoute (lastfm-1K).

Deux familles (comme MovieLens), adaptées aux séances :
1. RETRIEVAL — le latent prédit ẑ de la prochaine séance retrouve-t-il le futur ?
   - morceaux : banque des morceaux (chacun encodé comme séance-singleton via l'encodeur
     cible) ; Recall@K et rang médian des morceaux de la vraie prochaine séance.
   - séance   : plus proche voisin de ẑ parmi les vraies prochaines séances des users.
   Baselines : popularité, répétition-récence (derniers morceaux du user), répétition-fréquence
   (favoris du user), et pour la séance : répétition de la dernière séance du contexte.
2. PROBE — sonde linéaire h -> artistes de la prochaine séance (AUC macro) : le latent code-t-il
   le goût ?
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data_jepa import JepaEvalDataset, collate_eval, SESSION_CAP


# --------------------------------------------------------------------------- #
# Encodages
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _encode_sessions(encoder, sess_list, device, batch: int = 1024, cap: int = SESSION_CAP):
    """Encode une liste de séances (arrays d'idx morceau, taille variable) -> (N, d)."""
    out = []
    for i in range(0, len(sess_list), batch):
        chunk = sess_list[i:i + batch]
        maxL = max(max(len(s), 1) for s in chunk)
        tr = np.zeros((len(chunk), maxL), dtype=np.int64)
        tm = np.zeros((len(chunk), maxL), dtype=bool)
        for j, s in enumerate(chunk):
            s = s[-cap:] if len(s) > cap else s
            tr[j, :len(s)] = s
            tm[j, :len(s)] = True
        out.append(encoder(torch.as_tensor(tr).to(device),
                           torch.as_tensor(tm).to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def build_item_bank(model, n_tracks: int, device, batch: int = 2048) -> torch.Tensor:
    """Chaque morceau encodé comme séance-singleton (encodeur CIBLE) -> (n_tracks+1, d)."""
    singletons = [np.array([i], dtype=np.int64) for i in range(n_tracks + 1)]
    return _encode_sessions(model.target_session, singletons, device, batch=batch)


@torch.no_grad()
def encode_eval(model, ds, device, batch: int = 64):
    """-> (ẑ prédit, h contexte) à la dernière position réelle de chaque user, ordre du dataset."""
    loader = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_eval)
    Z, H = [], []
    for sess, tm, sm, _tg, _tgm in loader:
        sess, tm, sm = sess.to(device), tm.to(device), sm.to(device)
        B, M, L = sess.shape
        c = model.online_session(sess.reshape(-1, L), tm.reshape(-1, L)).reshape(B, M, -1)
        h = model.temporal(c, ~sm)
        zhat = model.predictor(h)
        last = sm.sum(1) - 1
        idx = torch.arange(B, device=device)
        Z.append(zhat[idx, last].cpu()); H.append(h[idx, last].cpu())
    return torch.cat(Z), torch.cat(H)


# --------------------------------------------------------------------------- #
# Baselines (rankings de morceaux, par user)
# --------------------------------------------------------------------------- #
def _popularity(sessions_df, n_tracks: int) -> np.ndarray:
    counts = np.zeros(n_tracks + 1, dtype=np.int64)
    for items in sessions_df["items"].values:
        np.add.at(counts, np.asarray(items, dtype=np.int64), 1)
    counts[0] = -1
    return counts


def _recency_rank(ctx_sessions) -> list[int]:
    """Morceaux du contexte, du plus récent au plus ancien, dédupliqués."""
    flat = np.concatenate(ctx_sessions) if ctx_sessions else np.zeros(0, dtype=np.int64)
    seen, out = set(), []
    for t in flat[::-1]:                                   # du plus récent au plus ancien
        t = int(t)
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def _frequency_rank(ctx_sessions) -> list[int]:
    """Morceaux du contexte triés par nombre d'écoutes décroissant."""
    flat = np.concatenate(ctx_sessions) if ctx_sessions else np.zeros(0, dtype=np.int64)
    if flat.size == 0:
        return []
    vals, cnts = np.unique(flat, return_counts=True)
    return vals[np.argsort(-cnts)].tolist()


# --------------------------------------------------------------------------- #
# 1. Retrieval morceaux
# --------------------------------------------------------------------------- #
@torch.no_grad()
def track_retrieval(zhat, item_bank, ds, pop_rank, ks=(10, 50, 100, 500),
                    device="cpu", user_batch: int = 128):
    bank = F.normalize(item_bank.to(device), dim=-1)       # (n+1, d)
    q = F.normalize(zhat.to(device), dim=-1)               # (U, d)
    U, maxk = q.shape[0], max(ks)
    hit = {k: 0 for k in ks}
    hit_pop = {k: 0 for k in ks}
    hit_rec = {k: 0 for k in ks}
    hit_frq = {k: 0 for k in ks}
    ranks = []
    pop_top = {k: set(pop_rank[:k].tolist()) for k in ks}

    for s in range(0, U, user_batch):
        sc = q[s:s + user_batch] @ bank.T                  # (b, n+1)
        sc[:, 0] = -1e9
        topk = sc.topk(maxk, dim=1).indices.cpu().numpy()
        for l in range(sc.shape[0]):
            u = s + l
            truth = set(int(i) for i in ds.target[u].tolist() if int(i) != 0)
            if not truth:
                continue
            ranked = topk[l]
            for k in ks:
                kt = set(int(x) for x in ranked[:k])
                if truth & kt:
                    hit[k] += 1
                if truth & pop_top[k]:
                    hit_pop[k] += 1
            rec = _recency_rank(ds.context[u]); frq = _frequency_rank(ds.context[u])
            for k in ks:
                if truth & set(rec[:k]):
                    hit_rec[k] += 1
                if truth & set(frq[:k]):
                    hit_frq[k] += 1
            # rang exact du meilleur morceau vrai (modèle)
            ti = torch.as_tensor(list(truth), device=device)
            best = sc[l, ti].max()
            ranks.append(int((sc[l] > best).sum().item()))
    n = len(ranks)
    return {
        "recall_model": {k: hit[k] / n for k in ks},
        "recall_pop": {k: hit_pop[k] / n for k in ks},
        "recall_repeat_recency": {k: hit_rec[k] / n for k in ks},
        "recall_repeat_freq": {k: hit_frq[k] / n for k in ks},
        "median_rank_model": float(np.median(ranks)), "n": n,
    }


# --------------------------------------------------------------------------- #
# 2. Retrieval séance (plus proche voisin entre users)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def session_retrieval(model, zhat, ds, device, ks=(1, 5, 10, 20)):
    bank = F.normalize(_encode_sessions(model.target_session, ds.target, device), dim=-1)
    q = F.normalize(zhat, dim=-1)
    U = q.shape[0]
    sims = q @ bank.T                                      # (U, U)
    ranks = (sims.argsort(dim=1, descending=True) == torch.arange(U)[:, None]).float().argmax(1)
    recall = {k: float((ranks < k).float().mean()) for k in ks}

    last_ctx = [c[-1] for c in ds.context]                 # répétition de la dernière séance
    qrep = F.normalize(_encode_sessions(model.target_session, last_ctx, device), dim=-1)
    ranks_r = (qrep @ bank.T).argsort(dim=1, descending=True)
    ranks_r = (ranks_r == torch.arange(U)[:, None]).float().argmax(1)
    recall_rep = {k: float((ranks_r < k).float().mean()) for k in ks}
    return {"recall_model": recall, "recall_repeat_last": recall_rep, "n": U}


# --------------------------------------------------------------------------- #
# 3. Probe artiste
# --------------------------------------------------------------------------- #
def probe_artist(H, ds, track2artist, n_train=0.8, top_artists=50):
    """Classif linéaire h -> artistes de la prochaine séance. AUC macro (test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from collections import Counter

    labels = []
    for t in ds.target:
        arts = set(int(track2artist[int(x)]) for x in t.tolist() if int(x) != 0)
        labels.append(arts)
    freq = Counter(a for s in labels for a in s if a != 0)
    keep = [a for a, _ in freq.most_common(top_artists)]
    Y = np.array([[1 if a in s else 0 for a in keep] for s in labels])

    X = H.numpy()
    cut = int(len(X) * n_train)
    aucs = []
    for j in range(Y.shape[1]):
        if Y[:cut, j].sum() == 0 or Y[cut:, j].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[:cut], Y[:cut, j])
        p = clf.predict_proba(X[cut:])[:, 1]
        aucs.append(roc_auc_score(Y[cut:, j], p))
    return float(np.mean(aucs)) if aucs else float("nan")


# --------------------------------------------------------------------------- #
# Suite complète
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_model(model, sessions_df, maps, device, eval_users: int | None = None,
                   seed: int = 0, ks_track=(10, 50, 100, 500), ks_sess=(1, 5, 10, 20)) -> dict:
    """Toute la suite d'éval pour UN modèle -> dict de métriques."""
    sub = sessions_df
    if eval_users and eval_users < len(sessions_df):
        sub = sessions_df.sample(eval_users, random_state=seed).reset_index(drop=True)
    ds = JepaEvalDataset(sub, split="test")
    zhat, H = encode_eval(model, ds, device)
    item_bank = build_item_bank(model, maps.n_tracks, device)
    pop_rank = np.argsort(-_popularity(sessions_df, maps.n_tracks))

    tr = track_retrieval(zhat, item_bank, ds, pop_rank, ks=ks_track, device=device)
    sr = session_retrieval(model, zhat, ds, device, ks=ks_sess)
    auc = probe_artist(H, ds, maps.track2artist)
    return {
        "recall_track": tr["recall_model"], "recall_pop": tr["recall_pop"],
        "recall_repeat_recency": tr["recall_repeat_recency"],
        "recall_repeat_freq": tr["recall_repeat_freq"],
        "median_rank_track": tr["median_rank_model"],
        "recall_session": sr["recall_model"], "recall_repeat_last": sr["recall_repeat_last"],
        "probe_artist_auc": auc, "n_eval": tr["n"],
    }
