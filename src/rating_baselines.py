"""Baselines de prédiction de note (la barre honnête à battre).

Le projet prédit désormais des NOTES masquées (personnalisées) plutôt que la
trajectoire temporelle. Une grosse part de la RMSE est triviale : `μ + biais_film
+ biais_user`. Ce module mesure exactement combien chaque baseline absorbe, pour
(a) fixer la barre que le set-transformer devra battre et (b) décider si le modèle
doit cibler la note BRUTE ou le RÉSIDU (note − biais).

Biais régularisés à la Koren (Netflix) :
    b_i = Σ_u (r_ui − μ) / (λ_i + n_i)
    b_u = Σ_i (r_ui − μ − b_i) / (λ_u + n_u)     (ordre : film d'abord, puis user)

Split « notes masquées » : pour chaque user, une fraction aléatoire de ses notes
est tenue en test, le reste sert à estimer μ / b_i / b_u. Pas de leave-last : le
signal temporel de MovieLens est du bruit (cf. mémoire projet).

Réutilisable à l'éval du modèle : `mask_split` + `fit_bias` donnent la même barre.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RATING_MIN, RATING_MAX = 0.5, 5.0
DEFAULT_PARQUET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "sequences.parquet",
)


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_long(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """`sequences.parquet` (une ligne/user, listes) -> table longue (user, item, rating)."""
    users = np.repeat(sequences_df["userId"].values, sequences_df["length"].values)
    items = np.concatenate(sequences_df["items"].values)
    ratings = np.concatenate(sequences_df["ratings"].values).astype(np.float32)
    return pd.DataFrame({"user": users, "item": items, "rating": ratings})


def mask_split(long: pd.DataFrame, frac: float = 0.2, seed: int = 0,
               min_ratings: int = 10):
    """Tient `frac` des notes de chaque user (≥ min_ratings) en test, reste en train.

    Masquage aléatoire intra-user (reproductible), analogue à la tâche du modèle.
    Les users trop courts (< min_ratings) restent entièrement en train.
    """
    rng = np.random.default_rng(seed)
    n = len(long)
    is_test = np.zeros(n, dtype=bool)
    # positions groupées par user (les listes viennent user par user, mais on ne
    # suppose rien : on regroupe explicitement via argsort stable)
    order = np.argsort(long["user"].values, kind="stable")
    users_sorted = long["user"].values[order]
    # bornes de chaque bloc user
    boundaries = np.flatnonzero(np.diff(users_sorted)) + 1
    blocks = np.split(order, boundaries)
    for idx in blocks:
        if len(idx) < min_ratings:
            continue
        k = int(round(len(idx) * frac))
        if k == 0:
            continue
        chosen = rng.choice(idx, size=k, replace=False)
        is_test[chosen] = True
    return long[~is_test].copy(), long[is_test].copy()


# --------------------------------------------------------------------------- #
# Modèle de biais
# --------------------------------------------------------------------------- #
def fit_bias(train: pd.DataFrame, lam_i: float = 10.0, lam_u: float = 10.0):
    """Estime μ, b_i, b_u régularisés sur le train. Renvoie (mu, b_i: Series, b_u: Series)."""
    mu = float(train["rating"].mean())
    # b_i : biais film régularisé
    g_i = train.groupby("item")["rating"]
    sum_i = g_i.sum() - mu * g_i.size()          # Σ(r − μ)
    b_i = sum_i / (lam_i + g_i.size())
    # résidu après μ + b_i, puis b_u régularisé
    resid = train["rating"].values - mu - train["item"].map(b_i).values
    tmp = pd.DataFrame({"user": train["user"].values, "resid": resid})
    g_u = tmp.groupby("user")["resid"]
    b_u = g_u.sum() / (lam_u + g_u.size())
    return mu, b_i, b_u


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def _rmse_mae(pred, true):
    err = np.clip(pred, RATING_MIN, RATING_MAX) - true
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def eval_baselines(train: pd.DataFrame, test: pd.DataFrame,
                   lam_i: float = 10.0, lam_u: float = 10.0) -> dict:
    """RMSE/MAE des baselines sur le test (paramètres estimés sur train)."""
    mu, b_i, b_u = fit_bias(train, lam_i, lam_u)
    true = test["rating"].values.astype(np.float32)

    bi = test["item"].map(b_i).fillna(0.0).values.astype(np.float32)   # item froid -> 0
    bu = test["user"].map(b_u).fillna(0.0).values.astype(np.float32)   # user froid -> 0

    preds = {
        "mu": np.full_like(true, mu),
        "mu+b_i": mu + bi,
        "mu+b_u": mu + bu,
        "mu+b_u+b_i": mu + bu + bi,
    }
    var = float(true.var())
    out = {"_meta": {"lam_i": lam_i, "lam_u": lam_u, "mu": mu,
                     "n_train": len(train), "n_test": len(test),
                     "test_std": float(true.std()), "test_var": var,
                     "cold_item_frac": float(test["item"].map(b_i).isna().mean())}}
    for name, p in preds.items():
        rmse, mae = _rmse_mae(p, true)
        out[name] = {"rmse": rmse, "mae": mae, "var_expl": 1.0 - rmse ** 2 / var}
    # std du résidu (échelle-cible si on prédit le résidu) — non clampé
    full_resid = true - (mu + bu + bi)
    out["_meta"]["resid_std"] = float(full_resid.std())
    return out


def _print_report(res: dict, lam_i: float, lam_u: float):
    m = res["_meta"]
    print(f"\n=== λ_i={lam_i}, λ_u={lam_u} | "
          f"n_train={m['n_train']:,} n_test={m['n_test']:,} | "
          f"μ={m['mu']:.4f} | test std={m['test_std']:.4f} ===")
    print(f"{'baseline':<14}{'RMSE':>9}{'MAE':>9}{'var_expl':>11}")
    for name in ("mu", "mu+b_i", "mu+b_u", "mu+b_u+b_i"):
        r = res[name]
        print(f"{name:<14}{r['rmse']:>9.4f}{r['mae']:>9.4f}{r['var_expl']*100:>10.1f}%")
    print(f"std du résidu (μ+b_u+b_i)   : {m['resid_std']:.4f}   "
          f"(échelle-cible si on prédit le résidu)")
    print(f"items froids en test        : {m['cold_item_frac']*100:.2f}%")


def main(parquet: str = DEFAULT_PARQUET, frac: float = 0.2, seed: int = 0):
    print(f"Chargement {parquet} ...")
    df = pd.read_parquet(parquet)
    long = load_long(df)
    print(f"Table longue : {len(long):,} notes | {long['user'].nunique():,} users | "
          f"{long['item'].nunique():,} films")
    train, test = mask_split(long, frac=frac, seed=seed)
    print(f"Split masqué (frac={frac}, seed={seed}) : "
          f"train={len(train):,}  test={len(test):,}")

    best = None
    for lam in (5.0, 10.0, 15.0, 25.0):
        res = eval_baselines(train, test, lam_i=lam, lam_u=lam)
        _print_report(res, lam, lam)
        full = res["mu+b_u+b_i"]["rmse"]
        if best is None or full < best[1]:
            best = (lam, full, res)

    lam, rmse, res = best
    print("\n" + "=" * 52)
    print(f"MEILLEUR MUR : μ+b_u+b_i @ λ={lam}  ->  RMSE={rmse:.4f}")
    print(f"std résidu    : {res['_meta']['resid_std']:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
