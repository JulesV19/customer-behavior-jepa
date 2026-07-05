"""Diagnostic fin du collapse observé à l'évaluation.

Objectif : comprendre le MÉCANISME avant de changer la loss.
On mesure, pour plusieurs jeux de représentations de chunk :
  - participation ratio (rang effectif) : ~1 = collapse dimensionnel, d = plein rang
  - std par dimension : VICReg voulait std >= 1 ; est-ce satisfait ?
  - corrélation moyenne |hors-diagonale| : dimensions corrélées => échec du terme covariance
  - cosine moyen BRUT vs CENTRÉ : vrai collapse, ou composante commune dominante ?

Comparaison clé : modèle ENTRAÎNÉ vs modèle NON-ENTRAÎNÉ (les Transformers sont
naturellement anisotropes à l'init ; on veut savoir si l'entraînement a empiré les choses).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data_prep import load_all
from src.data_jepa import to_chunks, CHUNK_SIZE
from src.jepa import TrajectoryJEPA
from src.train_jepa import load_model, _device


@torch.no_grad()
def collect_chunk_reps(encoder, sequences, device, n=5000, seed=0):
    chunks = []
    for items in sequences.sample(min(3000, len(sequences)), random_state=seed)["items"].values:
        for c in to_chunks(items, CHUNK_SIZE):
            chunks.append(c)
        if len(chunks) >= n:
            break
    chunks = torch.as_tensor(np.stack(chunks[:n]))
    return encoder(chunks.to(device)).cpu().float().numpy()   # (n, d)


def stats(R: np.ndarray) -> dict:
    N, d = R.shape
    Rc = R - R.mean(0, keepdims=True)
    C = (Rc.T @ Rc) / (N - 1)                       # covariance (d, d)
    eig = np.linalg.eigvalsh(C)
    eig = np.clip(eig, 0, None)
    pr = (eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12)   # participation ratio
    std = np.sqrt(np.clip(np.diag(C), 0, None))
    # corrélation moyenne hors-diagonale
    dinv = 1.0 / (std + 1e-8)
    corr = C * dinv[:, None] * dinv[None, :]
    offdiag = corr[~np.eye(d, dtype=bool)]
    # cosine moyen brut vs centré (sur un sous-échantillon de paires)
    def mean_cos(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        idx = np.random.default_rng(0).permutation(len(Xn))
        return float((Xn * Xn[idx]).sum(1).mean())
    return {
        "participation_ratio": float(pr),
        "eff_rank_pct": float(pr / d * 100),
        "std_min": float(std.min()), "std_med": float(np.median(std)), "std_max": float(std.max()),
        "mean_abs_offdiag_corr": float(np.abs(offdiag).mean()),
        "cos_raw": mean_cos(R),
        "cos_centered": mean_cos(Rc),
    }


def show(name, s):
    print(f"\n=== {name} ===")
    print(f"  participation ratio : {s['participation_ratio']:.2f} / {128}  "
          f"({s['eff_rank_pct']:.1f} % du plein rang)")
    print(f"  std/dim  min={s['std_min']:.3f}  med={s['std_med']:.3f}  max={s['std_max']:.3f}  "
          f"(VICReg voulait >= 1)")
    print(f"  |corr| hors-diag moy : {s['mean_abs_offdiag_corr']:.3f}  (0 = décorrélé, 1 = tout corrélé)")
    print(f"  cosine moyen  BRUT={s['cos_raw']:.3f}   CENTRÉ={s['cos_centered']:.3f}")


def main():
    device = _device()
    print("device:", device)
    sequences, genome, maps = load_all()

    trained = load_model(device)
    fresh = TrajectoryJEPA(maps.n_items, genome, d_model=128, nhead=4).to(device).eval()

    R_tgt = collect_chunk_reps(trained.target_chunk, sequences, device)
    R_on = collect_chunk_reps(trained.online_chunk, sequences, device)
    R_fresh = collect_chunk_reps(fresh.target_chunk, sequences, device)

    show("Cible EMA (ENTRAÎNÉ)", stats(R_tgt))
    show("Online (ENTRAÎNÉ)", stats(R_on))
    show("Cible (NON-ENTRAÎNÉ, référence init)", stats(R_fresh))

    print("\n--- Lecture ---")
    print("Si PR entraîné << PR non-entraîné -> l'entraînement A CAUSÉ le collapse dimensionnel.")
    print("Si std/dim ~1 MAIS |corr| élevé -> VICReg a tenu la variance mais PAS la décorrélation")
    print("   (le terme covariance, poids 1, était trop faible).")
    print("Si cos CENTRÉ << cos BRUT -> dominé par une composante commune (offset), pas full collapse.")


if __name__ == "__main__":
    main()
