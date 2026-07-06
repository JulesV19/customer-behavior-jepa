"""Mise en forme du dataset lastfm-1K pour le JEPA de trajectoires (grain = morceau).

Décisions actées (cf. discussion) :
- Grain d'item     : MORCEAU `(artist, track)`. Item = ID morceau + ID artiste (partagé).
- Filtrage         : garder les morceaux vus par >= MIN_TRACK_USERS users DISTINCTS.
                     (Le côté 'user' du k-core n'est jamais contraignant : tous les users
                     ont des milliers d'écoutes ⇒ le 5-core = simple filtre sur les items.)
- Séances          : découpage par SILENCES, nouvelle séance dès un écart >= TAU (1 h).
- Réécoutes        : GARDÉES (contenu pur, pas de feature 'déjà entendu').
- Ordre            : chronologique ; l'ordre intra-séance est un vrai signal (position gardée
                     côté modèle). Tout l'historique est conservé (fenêtre glissante à l'entraînement).

Sortie (dans lastfm/data/processed/, gitignoré) :
- sessions.parquet : 1 ligne/user — items (list[int] idx morceau), sessions (list[int] idx de
                     séance, 0..n_sessions-1), n_sessions, n_events. L'artiste de chaque item se
                     retrouve via maps.track2artist (pas stocké, pour rester léger).
- maps.pkl         : track2idx, artist2idx, track2artist (array idx→idx), idx2track, sizes.

Le module lit le brut par STREAMING par user (fichier trié par user), en 2 passes :
  pass 1 → support par morceau (nb de users distincts) ;  pass 2 → séances + indexation.
"""
from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lastfm_data import EVENTS, SEP_KEY, _parse_ts, human_seconds

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

TAU = 3600              # seuil de silence (s) = 1 h
MIN_TRACK_USERS = 5     # garder les morceaux vus par >= 5 users distincts
MIN_SESSIONS = 5        # garder les users avec >= 5 séances (contexte + val + test)
PAD_IDX = 0             # index réservé au padding (morceau ET artiste)


@dataclass
class LastfmMaps:
    track2idx: dict[str, int]        # 'artist ||| track' -> 1..n_tracks
    artist2idx: dict[str, int]       # 'artist' -> 1..n_artists
    idx2track: dict[int, str]
    track2artist: np.ndarray         # (n_tracks+1,) idx morceau -> idx artiste (0 = pad)
    n_tracks: int
    n_artists: int


# --------------------------------------------------------------------------- #
# Streaming par user (le fichier est trié par user)
# --------------------------------------------------------------------------- #
def _iter_users(usecols, names, chunksize=500_000, n_users=None):
    """Yield (user, sub_df) pour chaque user, en recollant les users à cheval sur 2 chunks."""
    cur, parts, done = None, [], 0
    reader = pd.read_csv(EVENTS, sep="\t", header=None, usecols=usecols, names=names,
                         dtype=str, na_values=[""], keep_default_na=False, chunksize=chunksize)
    for chunk in reader:
        for u, sub in chunk.groupby("user", sort=False):
            if cur is None or u == cur:
                parts.append(sub); cur = u
            else:
                yield cur, pd.concat(parts); done += 1
                if n_users and done >= n_users:
                    return
                parts, cur = [sub], u
    if parts and (not n_users or done < n_users):
        yield cur, pd.concat(parts)


def _keys(sub) -> pd.Series:
    """Clé morceau 'artist ||| track' pour un sous-dataframe (artist, track)."""
    return sub["artist"].fillna("?") + SEP_KEY + sub["track"].fillna("?")


# --------------------------------------------------------------------------- #
# Passe 1 : support par morceau (nb de users distincts)
# --------------------------------------------------------------------------- #
def track_support(chunksize=500_000, n_users=None) -> Counter:
    """Compte, pour chaque morceau, le nombre de users DISTINCTS qui l'ont écouté."""
    support = Counter()
    for _, sub in _iter_users([0, 3, 5], ["user", "artist", "track"], chunksize, n_users):
        support.update(pd.unique(_keys(sub)))          # +1 par user (clés dédupliquées)
    return support


def build_maps(support: Counter, min_users=MIN_TRACK_USERS) -> LastfmMaps:
    """Construit les tables d'indexation à partir des morceaux qui survivent au filtre."""
    kept = sorted(k for k, c in support.items() if c >= min_users)
    track2idx = {k: i + 1 for i, k in enumerate(kept)}          # 0 = pad
    idx2track = {i: k for k, i in track2idx.items()}
    artists = sorted({k.split(SEP_KEY)[0] for k in kept})
    artist2idx = {a: i + 1 for i, a in enumerate(artists)}
    track2artist = np.zeros(len(kept) + 1, dtype=np.int32)
    for k, i in track2idx.items():
        track2artist[i] = artist2idx[k.split(SEP_KEY)[0]]
    return LastfmMaps(track2idx, artist2idx, idx2track, track2artist,
                      len(kept), len(artists))


# --------------------------------------------------------------------------- #
# Passe 2 : séances + indexation par user
# --------------------------------------------------------------------------- #
def build_sessions(maps: LastfmMaps, tau=TAU, min_sessions=MIN_SESSIONS,
                   chunksize=500_000, n_users=None) -> pd.DataFrame:
    """Pour chaque user : filtre aux morceaux gardés, découpe en séances, indexe.

    Renvoie un DataFrame : userId, items (idx morceau), sessions (idx de séance),
    n_sessions, n_events. Users avec < min_sessions ignorés.
    """
    rows = []
    for u, sub in _iter_users([0, 1, 3, 5], ["user", "ts_raw", "artist", "track"],
                              chunksize, n_users):
        keys = _keys(sub)
        idx = keys.map(maps.track2idx)                  # NaN si morceau filtré
        keep = idx.notna().values
        if keep.sum() < 2:
            continue
        items = idx.values[keep].astype(np.int32)
        ts = _parse_ts(sub["ts_raw"])[keep]
        order = np.argsort(ts, kind="mergesort")        # tri chronologique croissant
        items, ts = items[order], ts[order]
        sess = np.zeros(len(ts), dtype=np.int32)
        sess[1:] = np.cumsum(np.diff(ts) >= tau)        # nouvelle séance dès un silence >= tau
        n_sess = int(sess[-1]) + 1
        if n_sess < min_sessions:
            continue
        rows.append({"userId": u, "items": items.tolist(), "sessions": sess.tolist(),
                     "n_sessions": n_sess, "n_events": len(items)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sauvegarde / chargement
# --------------------------------------------------------------------------- #
def save_all(sessions: pd.DataFrame, maps: LastfmMaps) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(PROCESSED / "sessions.parquet", index=False)
    with open(PROCESSED / "maps.pkl", "wb") as f:
        pickle.dump({"track2idx": maps.track2idx, "artist2idx": maps.artist2idx,
                     "idx2track": maps.idx2track, "track2artist": maps.track2artist,
                     "n_tracks": maps.n_tracks, "n_artists": maps.n_artists}, f)


def load_all() -> tuple[pd.DataFrame, LastfmMaps]:
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    with open(PROCESSED / "maps.pkl", "rb") as f:
        maps = LastfmMaps(**pickle.load(f))
    return sessions, maps


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(n_users=None) -> None:
    print("1/3  Passe 1 — support par morceau (users distincts)…")
    support = track_support(n_users=n_users)
    print(f"     {len(support):,} morceaux vus (avant filtre)")

    print(f"2/3  Filtrage (>= {MIN_TRACK_USERS} users) + indexation…")
    maps = build_maps(support)
    print(f"     {maps.n_tracks:,} morceaux gardés, {maps.n_artists:,} artistes")

    print("3/3  Passe 2 — séances par user…")
    sessions = build_sessions(maps, n_users=n_users)
    tot_sess = int(sessions["n_sessions"].sum())
    print(f"     {len(sessions):,} users | {sessions['n_events'].sum():,} events | "
          f"{tot_sess:,} séances | séances/user méd. {int(sessions['n_sessions'].median())}")

    save_all(sessions, maps)
    print(f"\nArtefacts -> {PROCESSED}/  (sessions.parquet, maps.pkl)")


if __name__ == "__main__":
    run()
