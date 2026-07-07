"""Évaluation du set-transformer de notes masquées.

Juge principal : RMSE/MAE sur le held-out TEST (même split que l'entraînement) du modèle
`clamp(μ+b_i+b_u+résidu_hat)` versus le MUR baseline `clamp(μ+b_i+b_u)`. Le modèle doit
passer sous le mur (~0.86, cf. `src/rating_baselines.py`) pour prouver qu'il apprend du goût.

Compléments :
- Cold-start : RMSE par bucket de popularité du film (là où le genome doit aider).
- Sonde de représentation : le vecteur [USER] prédit-il le profil genome moyen du user (R²) ?
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge

from .data_prep import load_all
from .data_ratings import (RATING_MIN, RATING_MAX, split_user_items,
                           RatingEvalDataset, collate)


def _item_train_counts(sequences_df, test_masks, n_items):
    """Popularité (nb de notes hors test) par film -> pour les buckets cold-start."""
    counts = np.zeros(n_items + 1, dtype=np.int64)
    for items, tm in zip(sequences_df["items"].values, test_masks):
        it = np.asarray(items)[~tm]
        np.add.at(counts, it, 1)
    return counts


@torch.no_grad()
def evaluate_model(model, mu, b_i, sequences_df, maps, device, seed: int = 0,
                   eval_users: int | None = None, batch_size: int = 128):
    """RMSE/MAE globales (modèle vs mur) + décomposition cold-start. Renvoie un dict."""
    if eval_users:
        sequences_df = sequences_df.iloc[:eval_users].reset_index(drop=True)
    test_masks = split_user_items(sequences_df, frac=0.2, seed=seed)
    counts = _item_train_counts(sequences_df, test_masks, maps.n_items)

    ds = RatingEvalDataset(sequences_df, test_masks, mu, b_i)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    # collecte par cible : erreurs modèle/baseline + popularité de l'item
    err_m, err_b, pops, abs_m = [], [], [], []
    for batch in loader:
        items = batch["items"].to(device)
        levels = batch["levels"].to(device)
        bias = batch["bias"].to(device)
        kpm = batch["key_padding_mask"].to(device)
        tmask = batch["target_mask"]
        base = batch["target_baseline"]
        true = batch["target_true"]

        resid, _ = model(items, levels, bias, kpm)
        resid = resid.cpu()
        pred = torch.clamp(base + resid, RATING_MIN, RATING_MAX)
        bpred = torch.clamp(base, RATING_MIN, RATING_MAX)
        m = tmask
        err_m.append(((pred - true) ** 2)[m].numpy())
        abs_m.append((pred - true).abs()[m].numpy())
        err_b.append(((bpred - true) ** 2)[m].numpy())
        pops.append(counts[batch["items"].numpy()[m.numpy()]])

    err_m = np.concatenate(err_m); err_b = np.concatenate(err_b)
    abs_m = np.concatenate(abs_m); pops = np.concatenate(pops)

    def rmse(x): return float(np.sqrt(x.mean()))
    out = {
        "n_targets": int(len(err_m)),
        "model_rmse": rmse(err_m), "model_mae": float(abs_m.mean()),
        "baseline_rmse": rmse(err_b),
        "gain": rmse(err_b) - rmse(err_m),
        "cold_start": {},
    }
    # buckets de popularité (quantiles), pour voir où le modèle gagne
    edges = [0, 20, 100, 500, np.inf]
    labels = ["≤20", "21-100", "101-500", ">500"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = (pops > lo) & (pops <= hi)
        if sel.sum() > 0:
            out["cold_start"][lab] = {"n": int(sel.sum()),
                                      "model_rmse": rmse(err_m[sel]),
                                      "baseline_rmse": rmse(err_b[sel])}
    return out


@torch.no_grad()
def _collect_user_vecs(model, sequences_df, test_masks, mu, b_i, device, batch_size=128):
    """Vecteur [USER] par user (contexte = pool train), aligné sur l'ordre du dataset."""
    ds = RatingEvalDataset(sequences_df, test_masks, mu, b_i)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    vecs = []
    for batch in loader:
        _, uv = model(batch["items"].to(device), batch["levels"].to(device),
                      batch["bias"].to(device), batch["key_padding_mask"].to(device))
        vecs.append(uv.cpu().numpy())
    return np.concatenate(vecs), ds


def genome_probe(model, mu, b_i, sequences_df, genome, maps, device, seed: int = 0,
                 n_users: int = 4000, alpha: float = 10.0):
    """Le vecteur [USER] prédit-il le profil genome moyen du user ? R² (split users)."""
    sub = sequences_df.iloc[:n_users].reset_index(drop=True)
    test_masks = split_user_items(sub, frac=0.2, seed=seed)
    X, ds = _collect_user_vecs(model, sub, test_masks, mu, b_i, device)

    # cible = genome moyen des films de contexte (train pool) de chaque user retenu
    genome = np.asarray(genome, dtype=np.float32)
    Y = np.stack([genome[ctx_items].mean(axis=0) for ctx_items, *_ in ds.samples])
    n = len(X)
    n_tr = int(n * 0.8)
    reg = Ridge(alpha=alpha).fit(X[:n_tr], Y[:n_tr])
    pred = reg.predict(X[n_tr:])
    y = Y[n_tr:]
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()
    return {"n_users": int(n), "genome_r2": float(1 - ss_res / ss_tot)}


def full_report(name: str = "ratings.pt", eval_users: int | None = None,
                probe_users: int = 4000, seed: int = 0):
    """Charge un checkpoint et imprime le rapport complet (juge + cold-start + sonde)."""
    from .train_ratings import load_model
    sequences, genome, maps = load_all()
    model, mu, b_i = load_model(name=name)
    device = next(model.parameters()).device.type

    res = evaluate_model(model, mu, b_i, sequences, maps, device,
                         seed=seed, eval_users=eval_users)
    print(f"\n=== {name} | {res['n_targets']:,} cibles ===")
    print(f"{'':<14}{'RMSE':>9}{'MAE':>9}")
    print(f"{'modèle':<14}{res['model_rmse']:>9.4f}{res['model_mae']:>9.4f}")
    print(f"{'mur μ+b_u+b_i':<14}{res['baseline_rmse']:>9.4f}")
    g = res["gain"]
    print(f"gain vs mur : {'+' if g>=0 else ''}{g:.4f}  "
          f"({'le modèle bat le mur' if g>0 else 'le modèle NE bat PAS le mur'})")
    print("\ncold-start (RMSE modèle / mur par popularité du film) :")
    for lab, d in res["cold_start"].items():
        print(f"  pop {lab:<9} n={d['n']:>8,} | modèle {d['model_rmse']:.4f} | mur {d['baseline_rmse']:.4f}")

    probe = genome_probe(model, mu, b_i, sequences, genome, maps, device,
                         seed=seed, n_users=probe_users)
    print(f"\nsonde genome (vecteur [USER] -> profil genome) : R²={probe['genome_r2']:.3f} "
          f"({probe['n_users']:,} users)")
    return {"eval": res, "probe": probe}


if __name__ == "__main__":
    full_report()
