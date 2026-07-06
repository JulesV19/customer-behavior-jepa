"""Chargement & investigation du dataset lastfm-1K (Òscar Celma, 2010).

Objectif de ce module : outiller le notebook `06_lastfm_investigation` pour
trancher, DONNÉES EN MAIN, deux décisions de mise en forme avant de basculer le
POC JEPA sur ce dataset :

  1. **Le grain d'item** : un « item » = un ARTISTE, un TRACK `(artist, track)`,
     ou un TRACK par MBID ? Compromis densité vs précision (cf. catalog_stats,
     kcore_survival, repeat_stats).
  2. **Le grain temporel / de session** : à quel écart couper une session
     d'écoute (cf. all_gaps, sessions_per_user, session_size_summary).

Le fichier brut est trié par user puis timestamp DÉCROISSANT ; on re-trie en
croissant. `track-mbid` est vide ~12 % du temps -> le grain 'track_mbid' perd
ces events, alors que 'track' (clé texte) les garde.

Volumétrie : 992 users, 19,15 M plays, 2,4 Go. `load_events(n_users=...)` lit
seulement les premiers users (le fichier est trié par user), pour rester
interactif. `n_users=None` = tout (lourd en RAM).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EVENTS = RAW / "userid-timestamp-artid-artname-traid-traname.tsv"
PROFILE = RAW / "userid-profile.tsv"

# Colonnes du fichier d'events (tab-separated, sans en-tête).
_COLS = {0: "user", 1: "ts_raw", 2: "artist_mbid", 3: "artist",
         4: "track_mbid", 5: "track"}
SEP_KEY = " ||| "        # séparateur pour la clé texte (artist, track)

GRAINS = ("artist", "track", "track_mbid")

# Seuils de session usuels (secondes) pour l'analyse temporelle.
TAUS: dict[str, int] = {"20 min": 1200, "1 h": 3600, "6 h": 21600, "1 jour": 86400}


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def _parse_ts(s: pd.Series) -> np.ndarray:
    """'2009-05-04T23:08:57Z' -> unix secondes (int64), via numpy datetime64.

    On coupe le 'Z' (fuseau uniforme, sans effet sur les écarts) et on laisse
    numpy parser l'ISO 8601 — plus robuste que pd.to_datetime tz-aware.
    """
    return s.str.slice(0, 19).values.astype("datetime64[s]").astype("int64")


def load_events(n_users: int | None = 200, chunksize: int = 500_000) -> pd.DataFrame:
    """Charge les events des `n_users` premiers users -> DataFrame tidy.

    Colonnes : user, ts (unix s), artist, track, track_mbid. Le fichier étant
    trié par user, on s'arrête dès qu'on dépasse le n-ième user. `n_users=None`
    lit tout le fichier (attention à la RAM : ~19 M lignes).
    """
    usecols = [0, 1, 3, 4, 5]
    names = ["user", "ts_raw", "artist", "track_mbid", "track"]
    reader = pd.read_csv(EVENTS, sep="\t", header=None, usecols=usecols,
                         names=names, dtype=str, chunksize=chunksize,
                         na_values=[""], keep_default_na=False)
    stop = f"user_{(n_users or 0) + 1:06d}"
    parts = []
    for chunk in reader:
        if n_users is not None:
            chunk = chunk[chunk["user"] < stop]
        parts.append(chunk)
        if n_users is not None and len(chunk) == 0:
            break
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = _parse_ts(df["ts_raw"])
    return df.drop(columns=["ts_raw"])


def load_profiles() -> pd.DataFrame:
    """Profils users (gender, age, country, signup)."""
    return pd.read_csv(PROFILE, sep="\t", skiprows=1,
                       names=["user", "gender", "age", "country", "registered"],
                       dtype=str)


# --------------------------------------------------------------------------- #
# Grain d'item
# --------------------------------------------------------------------------- #
def item_key(df: pd.DataFrame, grain: str) -> pd.Series:
    """Clé d'item selon le grain choisi. NaN = item indéfini à ce grain."""
    if grain == "artist":
        return df["artist"]
    if grain == "track_mbid":
        return df["track_mbid"]
    if grain == "track":                                   # clé texte (artist, track)
        return df["artist"].fillna("?") + SEP_KEY + df["track"].fillna("?")
    raise ValueError(f"grain inconnu : {grain!r} (attendus : {GRAINS})")


def catalog_stats(df: pd.DataFrame, grains=GRAINS) -> pd.DataFrame:
    """Pour chaque grain : taille de catalogue, densité, concentration, couverture.

    - n_items          : taille du vocabulaire à ce grain ;
    - plays/item méd.  : nb médian de plays par item (densité) ;
    - %_items_1play    : part du catalogue vue une SEULE fois (queue froide) ;
    - %_events_top1pct : part des plays captée par le 1 % d'items les + écoutés
                         (concentration : haut = quelques items dominent) ;
    - %_events_couverts: part des events où l'item EST défini à ce grain
                         (track_mbid perd les ~12 % sans MBID).
    """
    n_events = len(df)
    rows = []
    for grain in grains:
        key = item_key(df, grain)
        defined = key.notna()
        counts = key[defined].value_counts()
        head = int(np.ceil(len(counts) * 0.01))
        rows.append({
            "grain": grain,
            "n_items": int(len(counts)),
            "plays/item méd.": float(counts.median()),
            "plays/item moy.": float(counts.mean()),
            "%_items_1play": float((counts == 1).mean()),
            "%_events_top1pct": float(counts.nlargest(head).sum() / counts.sum()),
            "%_events_couverts": float(defined.mean()),
        })
    return pd.DataFrame(rows)


def kcore_survival(df: pd.DataFrame, grain: str, ks=(5, 10, 20),
                   max_iter: int = 30) -> pd.DataFrame:
    """Survie au k-core (sur paires DISTINCTES user-item, repeats collapsés).

    Combien d'users / items / interactions restent si on exige que chaque user
    ET chaque item aient >= k interactions distinctes ? Un grain trop fin
    s'effondre au k-core (catalogue non viable) ; c'est le test décisif.
    """
    base = pd.DataFrame({"user": df["user"].values, "item": item_key(df, grain).values})
    base = base.dropna().drop_duplicates()                 # 1 arête = 1 paire distincte
    rows = []
    for k in ks:
        d = base
        for _ in range(max_iter):
            n0 = len(d)
            uc = d["user"].value_counts(); d = d[d["user"].isin(uc.index[uc >= k])]
            ic = d["item"].value_counts(); d = d[d["item"].isin(ic.index[ic >= k])]
            if len(d) == n0:
                break
        rows.append({"k": k, "users": int(d["user"].nunique()),
                     "items": int(d["item"].nunique()), "paires": int(len(d))})
    return pd.DataFrame(rows)


def repeat_stats(df: pd.DataFrame, grain: str) -> dict:
    """Consommation répétée à ce grain (signal absent de MovieLens).

    - repeat_fraction  : part des plays portant sur un item DÉJÀ écouté par le
                         user (1 - paires_distinctes/plays) ;
    - médiane items distincts / user ;
    - médiane (unique/total) par user : proche de 1 = peu de replay, proche de
                         0 = beaucoup de réécoute.
    """
    d = pd.DataFrame({"user": df["user"].values, "item": item_key(df, grain).values}).dropna()
    total = len(d)
    per_user = d.groupby("user")["item"]
    n_uniq = per_user.nunique()
    n_tot = per_user.size()
    return {
        "grain": grain,
        "plays": int(total),
        "paires_distinctes": int(d.drop_duplicates().shape[0]),
        "repeat_fraction": float(1 - d.drop_duplicates().shape[0] / total),
        "items_distincts/user_méd.": float(n_uniq.median()),
        "unique/total_user_méd.": float((n_uniq / n_tot).median()),
    }


# --------------------------------------------------------------------------- #
# Grain temporel / sessions
# --------------------------------------------------------------------------- #
def user_timestamps(df: pd.DataFrame) -> list[np.ndarray]:
    """Liste des timestamps unix TRIÉS CROISSANT, un array par user."""
    return [np.sort(g["ts"].to_numpy()) for _, g in df.groupby("user", sort=False)]


def all_gaps(ts_list: list[np.ndarray]) -> np.ndarray:
    """Tous les écarts (secondes) entre events consécutifs, mis en commun."""
    out = [np.diff(t) for t in ts_list if t.size >= 2]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _session_sizes(t: np.ndarray, tau: int) -> np.ndarray:
    """Tailles des sessions d'un user : coupe dès qu'un écart >= tau."""
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    b = np.flatnonzero(np.diff(t) >= tau) + 1
    return np.diff(np.concatenate(([0], b, [t.size])))


def sessions_per_user(ts_list: list[np.ndarray], tau: int) -> np.ndarray:
    """Nombre de sessions (vrais 'moments') par user pour un seuil `tau`."""
    return np.array([len(_session_sizes(t, tau)) for t in ts_list])


def session_size_summary(ts_list: list[np.ndarray], taus=TAUS) -> pd.DataFrame:
    """Distribution des tailles de session et du nb de sessions/user par seuil."""
    rows = []
    for label, tau in taus.items():
        sizes = [_session_sizes(t, tau) for t in ts_list]
        s = np.concatenate(sizes) if sizes else np.zeros(0)
        spu = np.array([len(x) for x in sizes])
        rows.append({
            "seuil": label, "tau_s": tau,
            "sessions/user méd.": float(np.median(spu)) if spu.size else np.nan,
            "sessions/user moy.": float(spu.mean()) if spu.size else np.nan,
            "taille méd.": float(np.median(s)) if s.size else np.nan,
            "taille moy.": float(s.mean()) if s.size else np.nan,
            "taille p90": float(np.percentile(s, 90)) if s.size else np.nan,
        })
    return pd.DataFrame(rows)


def human_seconds(s: float) -> str:
    """Durée en secondes -> libellé court ('4 min', '2 j')."""
    for unit, sec in (("an", 31_557_600), ("j", 86_400), ("h", 3_600), ("min", 60)):
        if s >= sec:
            return f"{s / sec:.0f} {unit}"
    return f"{s:.0f} s"


# --------------------------------------------------------------------------- #
# Choix du seuil : caractéristiques des séances EN FONCTION du seuil
# --------------------------------------------------------------------------- #
def session_stats_vs_tau(ts_list: list[np.ndarray], taus) -> pd.DataFrame:
    """Comment les séances changent quand on fait varier le seuil de silence `tau`.

    Pour chaque `tau` (secondes), on re-découpe TOUS les users et on agrège :
      - n_sessions            : nombre total de séances ;
      - sessions/user_méd.    : nb médian de séances par user ;
      - taille_méd./moy./p90  : morceaux par séance ;
      - durée_méd_min         : durée médiane d'une séance (minutes) ;
      - %_séances_1morceau    : part des séances singleton (bruit si trop élevé) ;
      - écart_méd_@coupure_min : écart médian AUX frontières retenues (minutes) —
                                un seuil sain coupe à de gros écarts, donc ce chiffre
                                doit rester nettement au-dessus de `tau`.
    """
    rows = []
    for tau in taus:
        sizes, durs, spu, cut_gaps = [], [], [], []
        for t in ts_list:
            if t.size == 0:
                continue
            d = np.diff(t)
            b = np.flatnonzero(d >= tau) + 1
            edges = np.concatenate(([0], b, [t.size]))
            sizes.append(np.diff(edges))
            durs.append(t[edges[1:] - 1] - t[edges[:-1]])
            spu.append(len(edges) - 1)
            if b.size:
                cut_gaps.append(d[b - 1])
        s = np.concatenate(sizes) if sizes else np.zeros(0)
        du = np.concatenate(durs) if durs else np.zeros(0)
        cg = np.concatenate(cut_gaps) if cut_gaps else np.zeros(0)
        rows.append({
            "tau_s": int(tau),
            "seuil": human_seconds(tau),
            "n_sessions": int(s.size),
            "sessions/user_méd.": float(np.median(spu)) if spu else np.nan,
            "taille_méd.": float(np.median(s)) if s.size else np.nan,
            "taille_moy.": float(s.mean()) if s.size else np.nan,
            "taille_p90": float(np.percentile(s, 90)) if s.size else np.nan,
            "durée_méd_min": float(np.median(du) / 60) if du.size else np.nan,
            "%_séances_1morceau": float((s == 1).mean()) if s.size else np.nan,
            "écart_méd_@coupure_min": float(np.median(cg) / 60) if cg.size else np.nan,
        })
    return pd.DataFrame(rows)


def gap_log_hist(gaps: np.ndarray, n_bins: int = 60):
    """Histogramme des écarts en échelle LOG -> (centres_en_secondes, comptes).

    Sert à repérer le « creux » (vallée) entre le mode intra-séance (secondes-minutes)
    et le mode inter-séances (heures-jours). Les écarts nuls sont ignorés (log).
    """
    g = gaps[gaps > 0]
    counts, edges = np.histogram(np.log10(g), bins=n_bins)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2)
    return centers, counts


def suggest_tau(gaps: np.ndarray, lo: int = 300, hi: int = 21_600,
                n_bins: int = 80) -> float:
    """Seuil suggéré = fond de la vallée des écarts dans la fenêtre [lo, hi] secondes.

    Heuristique (à valider à l'œil sur l'histogramme) : le minimum de densité entre
    5 min et 6 h par défaut — là où « encore dans la séance » bascule vers « revenu
    plus tard ».
    """
    centers, counts = gap_log_hist(gaps, n_bins)
    m = (centers >= lo) & (centers <= hi)
    if not m.any():
        return float("nan")
    idx = np.where(m)[0]
    return float(centers[idx[np.argmin(counts[idx])]])
