"""Set-transformer pour la prédiction de notes masquées (résidu).

Architecture (décisions figées, cf. mémoire `movielens-rating-pivot`) :
- Un user = un SAC de films. On tokenise chaque film (identité + genome + note si VISIBLE).
  Les films masqués passent en rating-free (l'`ItemTokenizer` sort une note nulle) → le
  modèle connaît l'identité et le contenu du film, pas sa note.
- Encodeur = Transformer PLAT, SANS encodage positionnel (l'ordre est du bruit sur
  MovieLens), attention pleine, dropout=0 (même hygiène anti-triche que le JEPA).
- Un token [USER] appris : sa sortie contextualisée = le VECTEUR DE GOÛT (à sonder).
- Tête MLP par position : sortie du film -> RÉSIDU scalaire prédit (note − μ − b_i − b_u).

On réutilise l'`ItemTokenizer` du JEPA (id_emb + genome_proj + rating_emb + bias_proj).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .jepa import ItemTokenizer


class SetRatingModel(nn.Module):
    def __init__(self, n_items: int, genome, d_model: int = 128, nhead: int = 4,
                 layers: int = 4, head_hidden: int = 256):
        super().__init__()
        self.tokenizer = ItemTokenizer(n_items, genome, d_model, use_ratings=True)
        self.user_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # PAS d'encodage positionnel : le sac de films n'a pas d'ordre signifiant.
        # dropout=0 : cohérent avec le diagnostic d'effondrement du JEPA.
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1)
        )

    def forward(self, items, levels, bias, key_padding_mask):
        """items/levels/bias/key_padding_mask: (B, L). Renvoie (résidus (B, L), user_vec (B, d))."""
        tok = self.tokenizer(items, levels, bias)              # (B, L, d)
        B = tok.shape[0]
        cls = self.user_token.expand(B, -1, -1)                # (B, 1, d)
        x = torch.cat([cls, tok], dim=1)                       # (B, 1+L, d)
        # token [USER] jamais masqué -> colonne False en tête
        pad = torch.cat([key_padding_mask.new_zeros(B, 1), key_padding_mask], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad)        # (B, 1+L, d)
        user_vec = out[:, 0]                                   # vecteur de goût
        resid = self.head(out[:, 1:]).squeeze(-1)              # (B, L) résidu par film
        return resid, user_vec
