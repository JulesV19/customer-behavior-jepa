"""Mise en forme des données MovieLens-20M pour un modèle JEPA de reco séquentielle.

Chaque fonction fait UNE étape, pour pouvoir la dérouler et l'inspecter pas à pas
dans le notebook `notebooks/01_data_preparation.ipynb`.

Décisions figées (cf. discussion) :
- Signal            : toutes les interactions (implicite). Note + timestamp gardés en réserve.
- Ordre             : chronologique à plat, écarts de temps ignorés en v1.
- Filtrage          : on ne garde que les films couverts par le genome, puis 5-core itératif.
- Item              : ID (embedding appris plus tard) + vecteur genome (relevance continue [0,1]).
- Cap longueur      : 500 derniers films par user (troncature à gauche).
- Split             : leave-one-out (dernier=test, avant-dernier=val, reste=train), dérivé au chargement.

Convention d'indexation des items :
- L'index interne 0 est RÉSERVÉ au PADDING (aucun film réel).
- Les films réels vont de 1 à n_items.
- La matrice genome a n_items+1 lignes ; la ligne 0 (padding) est nulle.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
PROCESSED = ROOT / "data" / "processed"

MIN_CORE = 5        # seuil du k-core (users ET films >= 5 interactions)
MAX_SEQ_LEN = 500   # cap : on garde les 500 derniers films par user
PAD_IDX = 0         # index interne réservé au padding


# --------------------------------------------------------------------------- #
# 1. Chargement
# --------------------------------------------------------------------------- #
def load_ratings() -> pd.DataFrame:
    """Charge rating.csv en typant sobrement. timestamp -> unix seconds (int64)."""
    df = pd.read_csv(
        DATA / "rating.csv",
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        parse_dates=["timestamp"],
    )
    df["timestamp"] = (df["timestamp"].astype("int64") // 10**9).astype(np.int64)
    return df


def genome_movie_ids() -> np.ndarray:
    """Ensemble des movieId couverts par le tag genome."""
    return pd.read_csv(DATA / "genome_scores.csv", usecols=["movieId"])["movieId"].unique()


def load_genome_tag_count() -> int:
    """Nombre de tags du genome (dimension du vecteur de contenu)."""
    return int(pd.read_csv(DATA / "genome_tags.csv")["tagId"].max())


# --------------------------------------------------------------------------- #
# 2. Filtrage
# --------------------------------------------------------------------------- #
def filter_to_genome(df: pd.DataFrame, genome_ids: np.ndarray) -> pd.DataFrame:
    """Ne garde que les interactions sur des films ayant un vecteur genome."""
    return df[df["movieId"].isin(genome_ids)].copy()


def k_core(df: pd.DataFrame, k: int = MIN_CORE, max_iter: int = 20) -> pd.DataFrame:
    """Filtrage k-core itératif : users ET films avec >= k interactions.

    Itératif car retirer des films peut faire passer un user sous le seuil (et vice-versa).
    On boucle jusqu'à point fixe.
    """
    cur = df
    for _ in range(max_iter):
        n0 = len(cur)
        uc = cur["userId"].value_counts()
        cur = cur[cur["userId"].isin(uc.index[uc >= k])]
        ic = cur["movieId"].value_counts()
        cur = cur[cur["movieId"].isin(ic.index[ic >= k])]
        if len(cur) == n0:
            break
    return cur.copy()


# --------------------------------------------------------------------------- #
# 3. Id maps
# --------------------------------------------------------------------------- #
@dataclass
class IdMaps:
    user2idx: dict[int, int]
    movie2idx: dict[int, int]      # movieId -> index interne (1..n_items)
    idx2movie: dict[int, int]      # index interne -> movieId
    titles: dict[int, str]         # index interne -> titre du film
    n_users: int
    n_items: int                   # nb de films réels (hors padding)


def build_id_maps(df: pd.DataFrame) -> IdMaps:
    """Construit les tables d'indexation. Films indexés 1..n_items (0 = padding)."""
    users = np.sort(df["userId"].unique())
    movies = np.sort(df["movieId"].unique())
    user2idx = {int(u): i for i, u in enumerate(users)}
    movie2idx = {int(m): i + 1 for i, m in enumerate(movies)}  # +1 : 0 réservé au pad
    idx2movie = {i: m for m, i in movie2idx.items()}

    movie_df = pd.read_csv(DATA / "movie.csv", usecols=["movieId", "title"])
    title_by_movie = dict(zip(movie_df["movieId"], movie_df["title"]))
    titles = {idx: title_by_movie.get(m, "?") for idx, m in idx2movie.items()}

    return IdMaps(user2idx, movie2idx, idx2movie, titles, len(users), len(movies))


# --------------------------------------------------------------------------- #
# 4. Séquences
# --------------------------------------------------------------------------- #
def build_sequences(df: pd.DataFrame, maps: IdMaps,
                    max_len: int = MAX_SEQ_LEN) -> pd.DataFrame:
    """Une ligne par user : historique ordonné par timestamp, tronqué aux `max_len` derniers.

    Colonnes de sortie :
      userId, items (list[int index interne]), ratings (list[float]),
      timestamps (list[int]), length.
    ratings et timestamps sont gardés EN RÉSERVE (non utilisés en v1) pour éviter
    de retraiter le fichier de 690 Mo plus tard.
    """
    d = df.copy()
    d["item"] = d["movieId"].map(maps.movie2idx).astype(np.int32)
    # Tri stable par (user, timestamp) ; on garde l'ordre d'apparition pour départager les ex æquo.
    d = d.sort_values(["userId", "timestamp"], kind="mergesort")

    rows = []
    for uid, g in d.groupby("userId", sort=True):
        items = g["item"].to_numpy()
        ratings = g["rating"].to_numpy()
        ts = g["timestamp"].to_numpy()
        if len(items) > max_len:                 # troncature à gauche : on garde les + récents
            items, ratings, ts = items[-max_len:], ratings[-max_len:], ts[-max_len:]
        rows.append({
            "userId": int(uid),
            "items": items.tolist(),
            "ratings": ratings.tolist(),
            "timestamps": ts.tolist(),
            "length": len(items),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Matrice genome
# --------------------------------------------------------------------------- #
def build_genome_matrix(maps: IdMaps, n_tags: int) -> np.ndarray:
    """Matrice [n_items+1, n_tags] des relevances genome, indexée par item interne.

    Ligne 0 = padding (nulle). Chaque autre ligne = vecteur de pertinence [0,1] du film.
    Tous les films retenus ont un genome (filtrage en amont), donc pas de ligne manquante.
    """
    mat = np.zeros((maps.n_items + 1, n_tags), dtype=np.float32)
    gs = pd.read_csv(
        DATA / "genome_scores.csv",
        dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float32},
    )
    gs = gs[gs["movieId"].isin(maps.movie2idx)]
    rows = gs["movieId"].map(maps.movie2idx).to_numpy()      # index item interne
    cols = gs["tagId"].to_numpy() - 1                        # tagId 1..n_tags -> col 0..n_tags-1
    mat[rows, cols] = gs["relevance"].to_numpy()
    return mat


# --------------------------------------------------------------------------- #
# 6. Split leave-one-out (dérivé, pas stocké)
# --------------------------------------------------------------------------- #
def leave_one_out(items: list[int]) -> tuple[list[int], int, int]:
    """Découpe une séquence : (contexte_train, cible_val, cible_test).

    test  = dernier film, val = avant-dernier, train = tout le reste (le contexte).
    Suppose len(items) >= 3 (garanti par le 5-core).
    """
    return items[:-2], items[-2], items[-1]


# --------------------------------------------------------------------------- #
# 7. Sauvegarde / chargement des artefacts
# --------------------------------------------------------------------------- #
def save_all(sequences: pd.DataFrame, genome: np.ndarray, maps: IdMaps) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sequences.to_parquet(PROCESSED / "sequences.parquet", index=False)
    np.save(PROCESSED / "genome.npy", genome)
    # On sérialise un dict simple (pas l'objet dataclass) pour rester lisible
    # quel que soit le point d'entrée (module, notebook, script).
    with open(PROCESSED / "id_maps.pkl", "wb") as f:
        pickle.dump(maps.__dict__, f)


def load_all() -> tuple[pd.DataFrame, np.ndarray, IdMaps]:
    sequences = pd.read_parquet(PROCESSED / "sequences.parquet")
    genome = np.load(PROCESSED / "genome.npy")
    with open(PROCESSED / "id_maps.pkl", "rb") as f:
        maps = IdMaps(**pickle.load(f))
    return sequences, genome, maps


# --------------------------------------------------------------------------- #
# Orchestration complète
# --------------------------------------------------------------------------- #
def run() -> None:
    print("1/6  Chargement des ratings…")
    df = load_ratings()
    print(f"     {len(df):,} interactions, {df['userId'].nunique():,} users, "
          f"{df['movieId'].nunique():,} films")

    print("2/6  Filtrage aux films couverts par le genome…")
    df = filter_to_genome(df, genome_movie_ids())
    print(f"     {len(df):,} interactions, {df['movieId'].nunique():,} films")

    print("3/6  Filtrage 5-core itératif…")
    df = k_core(df, MIN_CORE)
    print(f"     {len(df):,} interactions, {df['userId'].nunique():,} users, "
          f"{df['movieId'].nunique():,} films")

    print("4/6  Construction des id maps…")
    maps = build_id_maps(df)
    print(f"     {maps.n_users:,} users, {maps.n_items:,} films (index 1..{maps.n_items})")

    print("5/6  Construction des séquences (cap = %d)…" % MAX_SEQ_LEN)
    sequences = build_sequences(df, maps, MAX_SEQ_LEN)
    print(f"     {len(sequences):,} séquences, longueur médiane "
          f"{int(sequences['length'].median())}")

    print("6/6  Construction de la matrice genome…")
    n_tags = load_genome_tag_count()
    genome = build_genome_matrix(maps, n_tags)
    print(f"     genome {genome.shape} (ligne 0 = padding)")

    save_all(sequences, genome, maps)
    print(f"\nArtefacts sauvegardés dans {PROCESSED}/")


if __name__ == "__main__":
    run()
