# Customer Behavior — JEPA de trajectoires de consommateurs

Preuve de concept : un **JEPA hiérarchique** capte-t-il la structure des trajectoires de
consommateurs (prédire les prochains films vus) sur **MovieLens-20M** ? Objectif =
*compréhension* des trajectoires, pas un recommandeur de production.

## Statut

- ✅ Mise en forme des données (notebook 01)
- ✅ Architecture JEPA (notebook 02)
- 🔄 Entraînement 15 epochs sur Colab GPU T4
- ⏭️ **Prochaine étape** : évaluation (`notebooks/03_training_evaluation.ipynb`) une fois
  `jepa.pt` récupéré depuis Colab.

## Structure

```
Data/                         CSV MovieLens bruts (gitignoré, ~900 Mo)
data/processed/               artefacts générés (gitignoré) : sequences.parquet, genome.npy, id_maps.pkl
src/
  data_prep.py                mise en forme -> artefacts (run: python -m src.data_prep)
  data_jepa.py                datasets niveau chunk (train + eval leave-last-chunk-out)
  jepa.py                     modèle : ChunkEmbedder, TemporalEncoder, Predictor, TrajectoryJEPA, jepa_loss
  train_jepa.py               entraînement + load_model (run: python -m src.train_jepa)
  evaluate_jepa.py            retrieval (films/chunks), baselines, probes, UMAP
notebooks/
  01_data_preparation.ipynb   déroulé explicable de la mise en forme
  02_jepa_architecture.ipynb  archi + passe avant + mini-entraînement de contrôle
  03_training_evaluation.ipynb entraînement rechargé + évaluation POC
scripts/                      analyses data-driven (écarts de temps, longueurs)
explore_data.ipynb            visualisation initiale des données
```

## Décisions figées (voir docstrings de `src/data_prep.py` et `src/jepa.py`)

**Données** : toutes les interactions (implicite), ordre chronologique à plat, écarts de temps
ignorés en v1. Filtrage : films avec vecteur genome → 5-core → **10 345 films / 138 493 users**.
Item = embedding d'ID **+ genome** (1128 tags). Cap 500 films/user. Split leave-one-out (niveau chunk).

**Architecture JEPA hiérarchique** : chunks de K=5 films (ensemble, ordre interne ignoré) →
encodeur de chunk dédié (Transformer + token [CHUNK]) → Transformer temporel causal → prédicteur
MLP → prédit la représentation du **prochain chunk**. Cible = **EMA** de l'encodeur de chunk
(valide car il s'entraîne côté contexte). Perte = `1 - cosine(ẑ, z_next)` + **VICReg**.
Petit modèle d=128, 4 têtes, 2 couches (~2,34M params). Régression JEPA pure (pas d'InfoNCE).

> ⚠️ Caveat data : les timestamps MovieLens sont des dates de **notation en rafales**, pas de
> visionnage. D'où les chunks (ordre bruité *dans* un chunk, signifiant *entre* chunks).
> Conséquence : le retrieval peut ne pas battre la popularité — ce n'est pas forcément un échec.

## Environnement

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m ipykernel install --user --name customer-behavior --display-name "Python (Customer Behavior)"
```
Notebooks : sélectionner le kernel **Python (Customer Behavior)**. Device auto : cuda → mps → cpu.

## Entraînement sur Colab (GPU)

1. Runtime → GPU T4.
2. `!git clone https://github.com/JulesV19/customer-behavior-jepa.git`
3. Copier `data/processed/` (3 fichiers, ~100 Mo) depuis Google Drive.
4. `!pip install -q pyarrow umap-learn` — **ne pas réinstaller torch** (CUDA déjà présent).
5. `from src.train_jepa import run; run(epochs=15, batch_size=256)`
6. Récupérer `data/processed/jepa.pt` + `train_history.json` (via Drive) pour l'évaluation locale.

## Lecture honnête des résultats

« Loss basse » ≠ « comprend ». Le verdict vient de l'évaluation :
- **retrieval vs popularité / répétition** : bat-il le trivial ?
- **probes genome (R²) / genres (AUC)** : l'espace latent code-t-il le contenu ?
- **anti-effondrement** : `pred_std` / `val_std` restent > 0 ?
