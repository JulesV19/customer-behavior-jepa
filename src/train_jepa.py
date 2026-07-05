"""Entraînement du JEPA de trajectoires.

Sauvegarde un checkpoint (poids + config) et l'historique des métriques, pour que
le notebook d'évaluation recharge le modèle sans avoir à ré-entraîner.

Défauts (non demandés à l'utilisateur, indiqués pour transparence) :
- optimiseur AdamW, lr=1e-3, weight_decay=1e-4
- warmup linéaire 5 % des steps puis décroissance cosine
- EMA de la cible = 0.996 (fixe)
- batch = 128 users, 15 epochs
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_prep import PROCESSED, load_all
from .data_jepa import JepaTrainDataset, JepaEvalDataset, collate_train, collate_eval, CHUNK_SIZE
from .jepa import TrajectoryJEPA, jepa_loss


def _device() -> str:
    if torch.cuda.is_available():          # Colab / GPU NVIDIA
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def _lr_factor(step: int, total: int, warmup: float = 0.05) -> float:
    w = max(1, int(total * warmup))
    if step < w:
        return step / w
    p = (step - w) / max(1, total - w)
    return 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def _collapse_on_val(model, val_loader, device, max_batches: int = 20):
    """Détecteur d'effondrement FIABLE : en mode EVAL (dropout coupé), on regarde les
    représentations de chunk (online). cos ~1.0 => effondrement ; std ~0 => effondrement.
    (L'ancien détecteur regardait le prédicteur, ce qui masquait le collapse de l'encodeur.)
    """
    model.eval()
    reps = []
    for i, (chunks, mask, _tgt, levels, bias) in enumerate(val_loader):
        chunks, mask = chunks.to(device), mask.to(device)
        K = chunks.shape[-1]
        valid = mask.reshape(-1)
        lv = levels.to(device).reshape(-1, K) if model.use_ratings else None
        bs = bias.to(device).reshape(-1, K) if model.use_ratings else None
        c = model.online_chunk(chunks.reshape(-1, K), lv, bs)[valid]
        reps.append(c.cpu())
        if i + 1 >= max_batches:
            break
    model.train()
    if not reps:
        return float("nan"), float("nan")
    R = torch.cat(reps).float()
    std = float(R.std(dim=0).mean())
    Rn = torch.nn.functional.normalize(R, dim=-1)
    perm = torch.randperm(R.shape[0])
    cos = float((Rn * Rn[perm]).sum(-1).mean())               # cosine moyen entre chunks au hasard
    return std, cos


def run(epochs: int = 15, batch_size: int = 128, d_model: int = 128, nhead: int = 4,
        chunk_layers: int = 2, temporal_layers: int = 2, pred_hidden: int = 256,
        use_ratings: bool = True, lr: float = 1e-3, ema: float = 0.996,
        subset: int | None = None, seed: int = 0, num_workers: int = 2):
    device = _device()
    torch.manual_seed(seed)
    print(f"device={device} | epochs={epochs} | batch={batch_size} | workers={num_workers}", flush=True)
    print(f"archi : d_model={d_model} nhead={nhead} chunk_layers={chunk_layers} "
          f"temporal_layers={temporal_layers} pred_hidden={pred_hidden} use_ratings={use_ratings}", flush=True)

    sequences, genome, maps = load_all()
    if subset:
        sequences = sequences.iloc[:subset].reset_index(drop=True)
        print(f"SUBSET actif : {len(sequences):,} users", flush=True)

    pin = device == "cuda"                        # pin_memory n'a de sens que pour CUDA
    train_ds = JepaTrainDataset(sequences, K=CHUNK_SIZE, min_chunks=4)
    val_ds = JepaEvalDataset(sequences, split="val", K=CHUNK_SIZE, min_chunks=3)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_train, drop_last=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_eval,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=num_workers > 0)
    print(f"train users={len(train_ds):,} | val users={len(val_ds):,} | "
          f"steps/epoch={len(train_loader):,}", flush=True)

    model = TrajectoryJEPA(maps.n_items, genome, d_model=d_model, nhead=nhead,
                           chunk_layers=chunk_layers, temporal_layers=temporal_layers,
                           pred_hidden=pred_hidden, ema=ema, use_ratings=use_ratings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)

    # Mixed precision (AMP) : activée UNIQUEMENT sur CUDA (~x1.5-2, libère de la mémoire).
    # Désactivée sur mps/cpu -> chemin identique au fp32 (autocast/scaler deviennent no-op).
    use_amp = device == "cuda"
    amp_dev = "cuda" if use_amp else "cpu"
    try:                                          # API récente (torch >= 2.3)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):           # fallback torch plus anciens
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP (mixed precision fp16) : {'ON' if use_amp else 'OFF'}", flush=True)

    history = []
    step = 0
    steps_per_epoch = len(train_loader)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        # accumulation en TENSEURS (sur le GPU, sans synchro) ; conversion float au log
        agg = {k: torch.zeros((), device=device) for k in ("loss", "inv", "var", "cov", "pred_std")}
        agg["n"] = 0
        for chunks, mask, levels, bias in train_loader:
            chunks, mask = chunks.to(device), mask.to(device)
            levels, bias = levels.to(device), bias.to(device)

            for g in opt.param_groups:
                g["lr"] = lr * _lr_factor(step, total_steps)
            opt.zero_grad()
            with torch.autocast(device_type=amp_dev, dtype=torch.float16, enabled=use_amp):
                c, z, zhat = model(chunks, mask, levels, bias)
                loss, logs = jepa_loss(c, z, zhat, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                              # dé-scale avant le clip de gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); model.update_target()

            for k in ("loss", "inv", "var", "cov", "pred_std"):
                agg[k] += logs[k]                            # add tenseur, pas de synchro
            agg["n"] += 1
            step += 1

            # Progression intra-epoch (les seules synchros : toutes les 100 steps)
            if agg["n"] % 100 == 0:
                rl = float(agg["loss"]) / agg["n"]
                ri = float(agg["inv"]) / agg["n"]
                print(f"  epoch {epoch+1:2d} | step {agg['n']:4d}/{steps_per_epoch} | "
                      f"loss {rl:.3f} | inv {ri:.4f} | pred_std {float(logs['pred_std']):.3f} | "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)

        tr = {k: float(agg[k]) / agg["n"] for k in ("loss", "inv", "var", "cov", "pred_std")}
        val_std, val_cos = _collapse_on_val(model, val_loader, device)
        dt = time.time() - t0
        rec = {"epoch": epoch + 1, **tr, "val_chunk_std": val_std, "val_chunk_cos": val_cos,
               "elapsed_s": round(dt, 1)}
        history.append(rec)
        # val_cos ~1.0 = EFFONDREMENT (à surveiller en priorité) ; doit rester bas (< ~0.5)
        print(f"epoch {epoch+1:2d}/{epochs} | loss {tr['loss']:.3f} | inv {tr['inv']:.4f} | "
              f"eval chunk std {val_std:.3f} cos {val_cos:.3f} | {dt/60:.1f} min", flush=True)

        # Checkpoint à chaque epoch (écrase) : évaluable même si on interrompt
        PROCESSED.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {"n_items": maps.n_items, "d_model": d_model, "nhead": nhead,
                       "chunk_layers": chunk_layers, "temporal_layers": temporal_layers,
                       "pred_hidden": pred_hidden, "use_ratings": use_ratings, "ema": ema},
            "epoch": epoch + 1,
        }
        torch.save(ckpt, PROCESSED / "jepa.pt")
        with open(PROCESSED / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nCheckpoint -> {PROCESSED/'jepa.pt'}  | historique -> train_history.json")
    return history


def load_model(device: str | None = None, name: str = "jepa.pt") -> TrajectoryJEPA:
    """Recharge un modèle entraîné depuis `data/processed/<name>` (défaut jepa.pt)."""
    device = device or _device()
    _, genome, _ = load_all()
    ckpt = torch.load(PROCESSED / name, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # fallback sur les défauts d'archi pour les anciens checkpoints (avant scale-up / ratings)
    model = TrajectoryJEPA(cfg["n_items"], genome, d_model=cfg["d_model"],
                           nhead=cfg["nhead"], chunk_layers=cfg.get("chunk_layers", 2),
                           temporal_layers=cfg.get("temporal_layers", 2),
                           pred_hidden=cfg.get("pred_hidden", 256),
                           use_ratings=cfg.get("use_ratings", False), ema=cfg["ema"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    run()
