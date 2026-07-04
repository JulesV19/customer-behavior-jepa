"""Distribution du nombre d'interactions (films) par utilisateur.

Sert à fixer le cap de longueur des séquences. On regarde la distribution
brute, puis l'effet du filtrage 5-core, et la couverture genome.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def describe(counts: np.ndarray, label: str):
    print(f"\n=== {label} ===")
    print(f"  users   : {len(counts):,}")
    print(f"  interactions totales : {int(counts.sum()):,}")
    pcts = [50, 75, 90, 95, 99, 99.9]
    print(f"  min={counts.min()}  max={counts.max()}  moyenne={counts.mean():.1f}")
    for p in pcts:
        print(f"  p{p:<5} : {np.percentile(counts, p):.0f} films")
    for cap in [50, 100, 200, 500]:
        cov = float(np.minimum(counts, cap).sum() / counts.sum())
        trunc = float((counts > cap).mean())
        print(f"  cap {cap:>4} -> {cov*100:5.1f} % des interactions gardees | "
              f"{trunc*100:5.1f} % des users tronques")


print("Chargement rating.csv (userId, movieId)…")
df = pd.read_csv(DATA / "rating.csv", usecols=["userId", "movieId"])
print(f"  {len(df):,} interactions | {df['userId'].nunique():,} users | "
      f"{df['movieId'].nunique():,} films")

# --- Distribution brute ---
raw = df.groupby("userId").size().to_numpy()
describe(raw, "BRUT (avant filtrage)")

# --- Filtrage 5-core itératif ---
print("\nApplication du 5-core itératif (users & films >= 5)…")
cur = df
for it in range(20):
    n0 = len(cur)
    uc = cur["userId"].value_counts()
    cur = cur[cur["userId"].isin(uc[uc >= 5].index)]
    ic = cur["movieId"].value_counts()
    cur = cur[cur["movieId"].isin(ic[ic >= 5].index)]
    if len(cur) == n0:
        print(f"  convergence à l'itération {it+1}")
        break
core = cur.groupby("userId").size().to_numpy()
print(f"  après 5-core : {len(cur):,} interactions | "
      f"{cur['userId'].nunique():,} users | {cur['movieId'].nunique():,} films")
describe(core, "APRES 5-core")

# --- Couverture genome ---
gs_movies = pd.read_csv(DATA / "genome_scores.csv", usecols=["movieId"])["movieId"].unique()
movies_core = set(cur["movieId"].unique())
inter = movies_core & set(gs_movies)
print(f"\n=== Couverture genome (sur films retenus 5-core) ===")
print(f"  films 5-core           : {len(movies_core):,}")
print(f"  films avec genome      : {len(inter):,} ({len(inter)/len(movies_core)*100:.1f} %)")
covered = cur[cur["movieId"].isin(inter)]
print(f"  interactions couvertes : {len(covered):,} / {len(cur):,} "
      f"({len(covered)/len(cur)*100:.1f} %)")

# --- Histogramme ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(np.log10(core), bins=70, color="#54A24B")
for cap in [50, 100, 200, 500]:
    x = np.log10(cap)
    ax.axvline(x, color="#F58518", ls="--", lw=1)
    ax.text(x, ax.get_ylim()[1]*0.95, str(cap), rotation=90,
            va="top", ha="right", color="#F58518", fontsize=9)
ax.set(title="Nb de films par user (après 5-core)",
       xlabel="log10(nb de films)", ylabel="Nb d'users")
fig.tight_layout()
fig.savefig(OUT / "length_distribution.png", dpi=120)
print(f"\nHistogramme -> {OUT / 'length_distribution.png'}")
