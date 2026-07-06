"""JEPA hiérarchique de trajectoires d'ÉCOUTE (dataset lastfm-1K, grain = morceau).

Adaptation du JEPA MovieLens aux séances d'écoute. Différences clés (cf. décisions) :
- **Item = morceau + artiste** : embedding d'ID du morceau + embedding de l'artiste
  (partagé entre les morceaux d'un même artiste → secours pour les morceaux rares).
  L'artiste se déduit du morceau via `track2artist` (buffer figé). Pas de notes.
- **Encodeur de séance à TAILLE VARIABLE + POSITION** : une séance est une suite ordonnée
  de morceaux (l'ordre d'écoute est un vrai signal, contrairement aux rafales MovieLens) →
  padding intra-séance + masque + encodage positionnel, résumé par un token [SESSION].
- **Cible = latent de la prochaine séance** produit par une copie EMA de l'encodeur de séance.
- Perte = MSE centrée (ẑ_t vs z_{t+1}) + VICReg (variance + covariance). dropout=0.

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
    """Morceau -> token : embedding du morceau + embedding de l'artiste (partagé).

    L'artiste est retrouvé en interne via `track2artist` (index morceau -> index artiste ;
    la ligne 0 = padding -> artiste 0 -> vecteur nul). On ne passe donc que l'index morceau.
    """

    def __init__(self, n_tracks: int, n_artists: int, track2artist, d_model: int,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.track_emb = nn.Embedding(n_tracks + 1, d_model, padding_idx=pad_idx)
        self.artist_emb = nn.Embedding(n_artists + 1, d_model, padding_idx=pad_idx)
        self.register_buffer("track2artist",
                             torch.as_tensor(track2artist, dtype=torch.long))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:      # (..., ) long
        artists = self.track2artist[tracks]                       # (...) index artiste
        return self.norm(self.track_emb(tracks) + self.artist_emb(artists))


class SessionEmbedder(nn.Module):
    """Encode une séance (suite ORDONNÉE de morceaux, taille variable) en un vecteur.

    Token [SESSION] + encodage positionnel (l'ordre compte) + padding intra-séance masqué.
    dropout=0 : sinon la variance VICReg peut être satisfaite par le bruit du dropout
    (l'encodeur triche, effondrement révélé en eval).
    """

    def __init__(self, tokenizer: ItemTokenizer, d_model: int, nhead: int, layers: int,
                 max_len: int, dropout: float = 0.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.session_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Embedding(max_len, d_model)                 # position intra-séance
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", norm_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, tracks: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        # tracks: (N, L) ; tok_mask: (N, L) True = morceau réel
        N, L = tracks.shape
        tok = self.tokenizer(tracks)                             # (N, L, d)
        tok = tok + self.pos(torch.arange(L, device=tracks.device))[None]
        cls = self.session_token.expand(N, -1, -1)              # (N, 1, d)
        x = torch.cat([cls, tok], dim=1)                        # (N, 1+L, d)
        # padding : le token [SESSION] est toujours gardé (False), puis ~tok_mask
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=tracks.device),
                         ~tok_mask], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad)
        return out[:, 0]                                        # sortie du token [SESSION]


class TemporalEncoder(nn.Module):
    """Transformer CAUSAL sur la suite des vecteurs de séance (l'ordre entre séances compte)."""

    def __init__(self, d_model: int, nhead: int, layers: int, max_sessions: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.pos = nn.Embedding(max_sessions, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", norm_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, c: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # c: (B, M, d) ; pad_mask: (B, M) True = séance de padding
        B, M, d = c.shape
        pos = self.pos(torch.arange(M, device=c.device))[None]
        causal = torch.triu(torch.ones(M, M, dtype=torch.bool, device=c.device), diagonal=1)
        return self.encoder(c + pos, mask=causal, src_key_padding_mask=pad_mask)


class Predictor(nn.Module):
    """MLP : état de contexte hₜ -> représentation prédite de la séance t+1."""

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
    def __init__(self, n_tracks: int, n_artists: int, track2artist, d_model: int = 128,
                 nhead: int = 4, session_layers: int = 2, temporal_layers: int = 2,
                 pred_hidden: int = 256, max_session_len: int = 30, max_sessions: int = 32,
                 ema: float = 0.996):
        super().__init__()
        tokenizer = ItemTokenizer(n_tracks, n_artists, track2artist, d_model)
        self.online_session = SessionEmbedder(tokenizer, d_model, nhead, session_layers,
                                              max_session_len)
        self.temporal = TemporalEncoder(d_model, nhead, temporal_layers, max_sessions)
        self.predictor = Predictor(d_model, pred_hidden)

        # Cible = copie EMA de l'encodeur de séance online (jamais entraînée par gradient)
        self.target_session = copy.deepcopy(self.online_session)
        for p in self.target_session.parameters():
            p.requires_grad_(False)
        self.ema = ema

    @torch.no_grad()
    def update_target(self) -> None:
        """θ_target ← ema·θ_target + (1-ema)·θ_online (params + buffers)."""
        for po, pt in zip(self.online_session.parameters(), self.target_session.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1.0 - self.ema)
        for bo, bt in zip(self.online_session.buffers(), self.target_session.buffers()):
            bt.copy_(bo)

    def forward(self, sessions: torch.Tensor, tok_mask: torch.Tensor,
                sess_mask: torch.Tensor):
        # sessions: (B, M, L) long ; tok_mask: (B, M, L) True=morceau réel ; sess_mask: (B, M)
        B, M, L = sessions.shape
        flat = sessions.reshape(B * M, L)
        flat_tm = tok_mask.reshape(B * M, L)
        c = self.online_session(flat, flat_tm).reshape(B, M, -1)          # online
        with torch.no_grad():
            z = self.target_session(flat, flat_tm).reshape(B, M, -1)      # cible EMA
        h = self.temporal(c, ~sess_mask)                                  # contexte causal
        zhat = self.predictor(h)                                          # ẑ_t -> séance t+1
        return c, z, zhat


# --------------------------------------------------------------------------- #
# Perte : MSE centrée + VICReg (repris du JEPA MovieLens v2, corrigé effondrement)
# --------------------------------------------------------------------------- #
def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_covariance(x: torch.Tensor, eps: float = 1e-4):
    """Termes VICReg sur un lot de représentations x: (P, d)."""
    std = torch.sqrt(x.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()                          # force std >= 1 par dim
    xc = x - x.mean(dim=0)
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    cov_loss = _off_diagonal(cov).pow(2).sum() / x.shape[1]      # décorrèle les dims
    return var_loss, cov_loss


def jepa_loss(c, z, zhat, sess_mask,
              lam_inv: float = 25.0, lam_var: float = 25.0, lam_cov: float = 1.0,
              center: bool = True):
    """Perte JEPA : invariance MSE centrée (ẑ_t vs z_{t+1}) + VICReg sur les repr online."""
    valid = sess_mask[:, :-1] & sess_mask[:, 1:]                 # paires t -> t+1 valides
    pred = zhat[:, :-1][valid]                                   # (P, d)
    tgt = z[:, 1:][valid]                                        # (P, d) détaché
    c_on = c[:, :-1][valid]                                      # (P, d) séance online

    pi, ti = pred, tgt
    if center:                                                  # matcher le résidu, pas l'offset
        pi = pred - pred.mean(dim=0, keepdim=True)
        ti = tgt - tgt.mean(dim=0, keepdim=True)
    inv = F.mse_loss(pi, ti)

    var_p, cov_p = variance_covariance(pred)
    var_c, cov_c = variance_covariance(c_on)
    var = 0.5 * (var_p + var_c)
    cov = 0.5 * (cov_p + cov_c)

    total = lam_inv * inv + lam_var * var + lam_cov * cov
    logs = {
        "loss": total.detach(), "inv": inv.detach(), "var": var.detach(),
        "cov": cov.detach(), "n_pairs": pred.shape[0],
        "pred_std": pred.std(dim=0).mean().detach(),
        "tgt_std": tgt.std(dim=0).mean().detach(),
    }
    return total, logs


def logs_to_float(logs: dict) -> dict:
    """Convertit les métriques (tenseurs) en floats. À n'appeler qu'au moment de logger."""
    return {k: (float(v) if torch.is_tensor(v) else v) for k, v in logs.items()}
