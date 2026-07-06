"""Contrôle VISUEL : le clustering k-means sur l'axe du temps retrouve-t-il les séances ?

On place un point par chanson selon son heure d'écoute, sur une fenêtre zoomée, et on
compare deux découpages du MÊME user :
  - haut  : k-means sur le temps (k = nombre de séances trouvées par la règle du silence) ;
  - bas   : règle du silence (nouvelle séance dès qu'un écart >= tau).
Les vrais silences (>= tau) sont tracés en pointillés verticaux. Un bon découpage
change de couleur PILE sur ces pointillés ; s'il coupe ailleurs (au milieu d'une rafale
de points serrés) ou garde la même couleur par-dessus un pointillé, il a raté la séance.

Usage : ../.venv/bin/python scripts/plot_session_clustering.py   (depuis lastfm/)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import lastfm_data as L

USER_IDX = 1          # quel user (1 = user_000001)
TAU = 3600            # seuil de silence (s) = 1 h
WINDOW_DAYS = 10      # largeur de la fenêtre zoomée
OUT = Path(__file__).resolve().parents[1] / "reports"
OUT.mkdir(exist_ok=True)


def sessions_by_gap(t, tau):
    """Id de séance par event (incrémenté à chaque silence >= tau)."""
    sess = np.zeros(len(t), dtype=int)
    sess[1:] = np.cumsum(np.diff(t) >= tau)
    return sess


def densest_window(t, days):
    """Début (unix s) de la fenêtre de `days` jours contenant le plus d'events."""
    w = days * 86400
    best_i, best_c = 0, 0
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] < t[i] + w:
            j += 1
        if j - i > best_c:
            best_c, best_i = j - i, i
    return t[best_i]


def main():
    df = L.load_events(n_users=USER_IDX)
    u = f"user_{USER_IDX:06d}"
    t = np.sort(df[df.user == u].ts.values.astype(float))

    gap_sess = sessions_by_gap(t, TAU)
    k = int(gap_sess.max()) + 1
    print(f"{u} : {len(t):,} chansons | {k} séances (règle silence >= {L.human_seconds(TAU)})")

    km = KMeans(n_clusters=k, n_init=2, random_state=0).fit(t.reshape(-1, 1))
    # réordonne les labels k-means dans l'ordre du temps
    order = np.argsort([t[km.labels_ == c].min() for c in range(k)])
    remap = {c: i for i, c in enumerate(order)}
    km_sess = np.array([remap[x] for x in km.labels_])

    # fenêtre zoomée = la plus dense
    start = densest_window(t, WINDOW_DAYS)
    m = (t >= start) & (t < start + WINDOW_DAYS * 86400)
    tw = t[m]
    dt = pd.to_datetime(tw, unit="s")
    gw, kw = gap_sess[m], km_sess[m]
    # silences réels dans la fenêtre (pour les pointillés)
    sil = pd.to_datetime(tw[1:][np.diff(tw) >= TAU], unit="s")
    print(f"fenêtre zoom : {len(tw)} chansons, {dt.min().date()} -> {dt.max().date()}, "
          f"{gw.max()-gw.min()+1} séances, {len(sil)} silences")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, lab, title in [(a1, kw, "k-means sur le temps"),
                           (a2, gw, f"règle du silence (>= {L.human_seconds(TAU)})")]:
        rel = pd.Series(lab).rank(method="dense").astype(int).values - 1
        ax.scatter(dt, np.zeros_like(tw) + (rel % 20), c=rel % 20, cmap="tab20",
                   s=18, alpha=0.9)
        for s in sil:
            ax.axvline(s, color="crimson", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(f"{title}  —  {len(np.unique(lab))} groupes dans la fenêtre")
        ax.set_yticks([])
        ax.set_ylabel("groupe (couleur)")
    a2.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    fig.suptitle(f"{u} — un point = une chanson ; pointillés rouges = vrais silences >= "
                 f"{L.human_seconds(TAU)}", fontsize=11)
    fig.tight_layout()
    out = OUT / f"session_clustering_user{USER_IDX}.png"
    fig.savefig(out, dpi=130)
    print("figure ->", out)


if __name__ == "__main__":
    main()
