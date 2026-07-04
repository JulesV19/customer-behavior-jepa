"""Architecture JEPA hiérarchique pour la prédiction de trajectoires de consommateurs.

Idée (cf. discussion) : on découpe l'historique d'un user en CHUNKS de K films.
- Un encodeur de chunk (dédié) transforme chaque chunk (un ENSEMBLE de K films, sans
  ordre interne) en un vecteur, via un petit Transformer + un token [CHUNK].
- Un Transformer temporel CAUSAL lit la suite des vecteurs de chunk du passé.
- Un prédicteur MLP prédit, à chaque position, la représentation du CHUNK SUIVANT.
- La cible est produite par une copie EMA de l'encodeur de chunk (stop-gradient).
  L'EMA est valide car l'encodeur de chunk reçoit du gradient côté contexte.
- Perte = régression cosine (ẑ vs z) + VICReg (variance + covariance) anti-effondrement.

Tout est prédit DANS l'espace des représentations : jamais de reconstruction d'ID.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Briques
# --------------------------------------------------------------------------- #
class ItemTokenizer(nn.Module):
    """Film -> token : embedding d'ID appris + projection du vecteur genome (fusion additive)."""

    def __init__(self, n_items: int, genome, d_model: int, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.id_emb = nn.Embedding(n_items + 1, d_model, padding_idx=pad_idx)
        genome_t = torch.as_tensor(genome, dtype=torch.float32)
        self.register_buffer("genome", genome_t)              # (n_items+1, n_tags), figé
        self.genome_proj = nn.Linear(genome_t.shape[1], d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, items: torch.Tensor) -> torch.Tensor:   # items: (...,) long
        tok = self.id_emb(items) + self.genome_proj(self.genome[items])
        return self.norm(tok)


class ChunkEmbedder(nn.Module):
    """Encode un ENSEMBLE de K films en un vecteur, via un token [CHUNK].

    Pas d'encodage positionnel : les K films d'un chunk forment un ensemble
    (cohérent avec l'artefact des rafales de notation).
    """

    def __init__(self, tokenizer: ItemTokenizer, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.chunk_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, chunk_items: torch.Tensor) -> torch.Tensor:  # (N, K) long
        tok = self.tokenizer(chunk_items)                          # (N, K, d)
        cls = self.chunk_token.expand(tok.shape[0], -1, -1)        # (N, 1, d)
        x = torch.cat([cls, tok], dim=1)                           # (N, 1+K, d)
        out = self.encoder(x)                                      # attention pleine
        return out[:, 0]                                           # sortie du token [CHUNK]


class TemporalEncoder(nn.Module):
    """Transformer CAUSAL sur la suite des vecteurs de chunk (l'ordre entre chunks compte)."""

    def __init__(self, d_model: int, nhead: int, layers: int, max_chunks: int = 128):
        super().__init__()
        self.pos = nn.Embedding(max_chunks, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, c: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # c: (B, M, d) ; pad_mask: (B, M) True = position de padding
        B, M, d = c.shape
        pos = self.pos(torch.arange(M, device=c.device))[None]     # (1, M, d)
        # masque causal booléen (True = interdit), même type que pad_mask
        causal = torch.triu(torch.ones(M, M, dtype=torch.bool, device=c.device), diagonal=1)
        return self.encoder(c + pos, mask=causal, src_key_padding_mask=pad_mask)


class Predictor(nn.Module):
    """MLP : état de contexte hₜ -> représentation prédite du chunk t+1."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# --------------------------------------------------------------------------- #
# Modèle complet
# --------------------------------------------------------------------------- #
class TrajectoryJEPA(nn.Module):
    def __init__(self, n_items: int, genome, d_model: int = 128, nhead: int = 4,
                 chunk_layers: int = 2, temporal_layers: int = 2, pred_hidden: int = 256,
                 max_chunks: int = 128, ema: float = 0.996):
        super().__init__()
        tokenizer = ItemTokenizer(n_items, genome, d_model)
        self.online_chunk = ChunkEmbedder(tokenizer, d_model, nhead, chunk_layers)
        self.temporal = TemporalEncoder(d_model, nhead, temporal_layers, max_chunks)
        self.predictor = Predictor(d_model, pred_hidden)

        # Cible = copie EMA de l'encodeur de chunk online (jamais entraînée par gradient)
        self.target_chunk = copy.deepcopy(self.online_chunk)
        for p in self.target_chunk.parameters():
            p.requires_grad_(False)
        self.ema = ema

    @torch.no_grad()
    def update_target(self) -> None:
        """Met à jour la cible : θ_target ← ema·θ_target + (1-ema)·θ_online."""
        for po, pt in zip(self.online_chunk.parameters(), self.target_chunk.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1.0 - self.ema)
        for bo, bt in zip(self.online_chunk.buffers(), self.target_chunk.buffers()):
            bt.copy_(bo)

    def forward(self, chunks: torch.Tensor, chunk_mask: torch.Tensor):
        # chunks: (B, M, K) long ; chunk_mask: (B, M) True = chunk réel
        B, M, K = chunks.shape
        pad_mask = ~chunk_mask
        flat = chunks.reshape(B * M, K)

        c = self.online_chunk(flat).reshape(B, M, -1)          # (B, M, d) online, avec gradient
        with torch.no_grad():
            z = self.target_chunk(flat).reshape(B, M, -1)      # (B, M, d) cible EMA, sans gradient
        h = self.temporal(c, pad_mask)                         # (B, M, d) contexte causal
        zhat = self.predictor(h)                               # (B, M, d) : ẑ_t prédit le chunk t+1
        return c, z, zhat


# --------------------------------------------------------------------------- #
# Perte : régression cosine + VICReg
# --------------------------------------------------------------------------- #
def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_covariance(x: torch.Tensor, eps: float = 1e-4):
    """Termes VICReg sur un lot de représentations x: (P, d)."""
    std = torch.sqrt(x.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()                        # force std >= 1 par dimension
    xc = x - x.mean(dim=0)
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    cov_loss = _off_diagonal(cov).pow(2).sum() / x.shape[1]    # décorrèle les dimensions
    return var_loss, cov_loss


def jepa_loss(c, z, zhat, chunk_mask,
              lam_inv: float = 25.0, lam_var: float = 25.0, lam_cov: float = 1.0):
    """Perte JEPA.

    - invariance : 1 - cosine(ẑ_t, z_{t+1})  (prédire la représentation du prochain chunk)
    - variance + covariance (VICReg) sur les représentations ONLINE (ẑ et chunk c),
      pour empêcher l'effondrement. La cible z est détachée (EMA), donc hors gradient.
    """
    valid = chunk_mask[:, :-1] & chunk_mask[:, 1:]             # paires t -> t+1 valides
    pred = zhat[:, :-1][valid]                                 # (P, d)
    tgt = z[:, 1:][valid]                                      # (P, d) détaché
    c_on = c[:, :-1][valid]                                    # (P, d) chunk online

    inv = (1.0 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
    var_p, cov_p = variance_covariance(pred)
    var_c, cov_c = variance_covariance(c_on)
    var = 0.5 * (var_p + var_c)
    cov = 0.5 * (cov_p + cov_c)

    total = lam_inv * inv + lam_var * var + lam_cov * cov
    logs = {
        "loss": float(total.detach()),
        "inv": float(inv.detach()),
        "var": float(var.detach()),
        "cov": float(cov.detach()),
        "n_pairs": int(pred.shape[0]),
        # métriques d'effondrement (à surveiller) :
        "pred_std": float(pred.std(dim=0).mean().detach()),   # ~0 => effondrement
        "tgt_std": float(tgt.std(dim=0).mean().detach()),
    }
    return total, logs
