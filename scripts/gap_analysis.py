"""Distribution des écarts de temps entre deux visionnages consécutifs d'un même user.

Sert à décider empiriquement s'il faut découper les trajectoires en sessions,
et avec quel seuil d'écart temporel.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "Data"
OUT = Path(__file__).resolve().parent.parent / "reports"
OUT.mkdir(exist_ok=True)

print("Chargement de rating.csv (userId + timestamp)…")
df = pd.read_csv(
    DATA / "rating.csv",
    usecols=["userId", "timestamp"],
    parse_dates=["timestamp"],
)
print(f"  {len(df):,} interactions | {df['userId'].nunique():,} users")

print("Tri par (user, timestamp) et calcul des écarts consécutifs…")
df = df.sort_values(["userId", "timestamp"], kind="mergesort")
# Écart avec l'interaction précédente DU MÊME user (NaT à la 1re de chaque user)
delta = df["timestamp"].diff()
same_user = df["userId"].eq(df["userId"].shift())
gaps = delta[same_user].dt.total_seconds().to_numpy()
gaps = gaps[gaps >= 0]  # sécurité
gaps_h = gaps / 3600.0
gaps_d = gaps / 86400.0

print(f"\n{len(gaps):,} écarts calculés (interactions - users)\n")

# --- Percentiles ---
pcts = [10, 25, 50, 75, 90, 95, 99, 99.9]
print("Percentiles des écarts :")
print(f"{'p':>6} | {'heures':>12} | {'jours':>12}")
print("-" * 36)
for p in pcts:
    vh = np.percentile(gaps_h, p)
    print(f"{p:>6} | {vh:>12.2f} | {vh/24:>12.2f}")

# --- Fraction au-dessus de seuils candidats de session ---
print("\nFraction des écarts DÉPASSANT un seuil (= nouvelle session) :")
thresholds = {
    "1 heure": 1, "6 heures": 6, "12 heures": 12, "24 heures": 24,
    "3 jours": 72, "7 jours": 168, "30 jours": 720, "90 jours": 2160,
}
for name, h in thresholds.items():
    frac = float((gaps_h > h).mean())
    print(f"  > {name:<9} : {frac*100:6.2f} %  -> {int((gaps_h > h).sum()):>12,} coupures")

# --- Histogramme (échelle log) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pos = gaps_h[gaps_h > 0]
logh = np.log10(pos)
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(logh, bins=80, color="#4C78A8")
# Repères verticaux lisibles
marks = {"1h": 1, "1j": 24, "1sem": 168, "1mois": 720, "1an": 8760}
for label, h in marks.items():
    x = np.log10(h)
    ax.axvline(x, color="#F58518", ls="--", lw=1)
    ax.text(x, ax.get_ylim()[1]*0.95, label, rotation=90,
            va="top", ha="right", color="#F58518", fontsize=9)
ax.set(title="Distribution des écarts entre deux visionnages consécutifs (même user)",
       xlabel="log10(écart en heures)", ylabel="Nb d'écarts")
fig.tight_layout()
fig.savefig(OUT / "gap_distribution.png", dpi=120)
print(f"\nHistogramme sauvegardé -> {OUT / 'gap_distribution.png'}")
