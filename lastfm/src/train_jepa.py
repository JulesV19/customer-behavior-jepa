"""Entraînement du JEPA d'écoute (lastfm-1K). Structure calquée sur la version MovieLens.

Défauts (pour transparence) : AdamW lr=1e-3 wd=1e-4 ; warmup 5 % puis cosine ; EMA 0.996 ;
batch=128 tuiles, 15 epochs ; AMP fp16 UNIQUEMENT sur CUDA (no-op sur mps/cpu).
Checkpoint (poids + config) + historique sauvés à chaque epoch dans data/processed/.
"""
from __future__ import annotations

import json
import math
import time

import torch
from torch.utils.data import DataLoader

from .lastfm_data_prep import PROCESSED, load_all
from .data_jepa import (JepaTrainDataset, JepaEvalDataset, collate_train, collate_eval,
                        WINDOW, SESSION_CAP)
from .jepa import TrajectoryJEPA, jepa_loss


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
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
    """Détecteur d'effondrement (mode EVAL) : sur les repr de séance online (contexte).

    cos ~1.0 => effondrement ; std ~0 => effondrement. Doit rester bas (cos < ~0.5).
    """
    model.eval()
    reps = []
    for i, (sess, tm, sm, _tg, _tgm) in enumerate(val_loader):
        sess, tm, sm = sess.to(device), tm.to(device), sm.to(device)
        L = sess.shape[-1]
        c = model.online_session(sess.reshape(-1, L), tm.reshape(-1, L))[sm.reshape(-1)]
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
    cos = float((Rn * Rn[perm]).sum(-1).mean())
    return std, cos


def run(epochs: int = 15, batch_size: int = 128, d_model: int = 128, nhead: int = 4,
        session_layers: int = 2, temporal_layers: int = 2, pred_hidden: int = 256,
        lr: float = 1e-3, ema: float = 0.996, subset: int | None = None,
        seed: int = 0, num_workers: int = 2):
    device = _device()
    torch.manual_seed(seed)
    print(f"device={device} | epochs={epochs} | batch={batch_size} | workers={num_workers}", flush=True)
    print(f"archi : d_model={d_model} nhead={nhead} session_layers={session_layers} "
          f"temporal_layers={temporal_layers} pred_hidden={pred_hidden} "
          f"window={WINDOW} cap={SESSION_CAP}", flush=True)

    sessions, maps = load_all()
    if subset:
        sessions = sessions.iloc[:subset].reset_index(drop=True)
        print(f"SUBSET actif : {len(sessions):,} users", flush=True)

    pin = device == "cuda"
    train_ds = JepaTrainDataset(sessions)
    val_ds = JepaEvalDataset(sessions, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_train, drop_last=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_eval,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=num_workers > 0)
    print(f"train tuiles={len(train_ds):,} | val users={len(val_ds):,} | "
          f"steps/epoch={len(train_loader):,}", flush=True)

    model = TrajectoryJEPA(maps.n_tracks, maps.n_artists, maps.track2artist, d_model=d_model,
                           nhead=nhead, session_layers=session_layers,
                           temporal_layers=temporal_layers, pred_hidden=pred_hidden,
                           max_session_len=SESSION_CAP, max_sessions=WINDOW, ema=ema).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)

    use_amp = device == "cuda"
    amp_dev = "cuda" if use_amp else "cpu"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP (mixed precision fp16) : {'ON' if use_amp else 'OFF'}", flush=True)

    history = []
    step = 0
    steps_per_epoch = len(train_loader)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        agg = {k: torch.zeros((), device=device) for k in ("loss", "inv", "var", "cov", "pred_std")}
        agg["n"] = 0
        for sess, tm, sm in train_loader:
            sess, tm, sm = sess.to(device), tm.to(device), sm.to(device)
            for g in opt.param_groups:
                g["lr"] = lr * _lr_factor(step, total_steps)
            opt.zero_grad()
            with torch.autocast(device_type=amp_dev, dtype=torch.float16, enabled=use_amp):
                c, z, zhat = model(sess, tm, sm)
                loss, logs = jepa_loss(c, z, zhat, sm)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); model.update_target()

            for k in ("loss", "inv", "var", "cov", "pred_std"):
                agg[k] += logs[k]
            agg["n"] += 1
            step += 1
            if agg["n"] % 100 == 0:
                print(f"  epoch {epoch+1:2d} | step {agg['n']:4d}/{steps_per_epoch} | "
                      f"loss {float(agg['loss'])/agg['n']:.3f} | "
                      f"inv {float(agg['inv'])/agg['n']:.4f} | "
                      f"pred_std {float(logs['pred_std']):.3f} | {(time.time()-t0)/60:.1f} min", flush=True)

        tr = {k: float(agg[k]) / agg["n"] for k in ("loss", "inv", "var", "cov", "pred_std")}
        val_std, val_cos = _collapse_on_val(model, val_loader, device)
        dt = time.time() - t0
        history.append({"epoch": epoch + 1, **tr, "val_session_std": val_std,
                        "val_session_cos": val_cos, "elapsed_s": round(dt, 1)})
        print(f"epoch {epoch+1:2d}/{epochs} | loss {tr['loss']:.3f} | inv {tr['inv']:.4f} | "
              f"eval séance std {val_std:.3f} cos {val_cos:.3f} | {dt/60:.1f} min", flush=True)

        PROCESSED.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {"n_tracks": maps.n_tracks, "n_artists": maps.n_artists, "d_model": d_model,
                       "nhead": nhead, "session_layers": session_layers,
                       "temporal_layers": temporal_layers, "pred_hidden": pred_hidden,
                       "max_session_len": SESSION_CAP, "max_sessions": WINDOW, "ema": ema},
            "epoch": epoch + 1,
        }
        torch.save(ckpt, PROCESSED / "jepa.pt")
        with open(PROCESSED / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nCheckpoint -> {PROCESSED/'jepa.pt'} | historique -> train_history.json")
    return history


def load_model(device: str | None = None, name: str = "jepa.pt") -> TrajectoryJEPA:
    """Recharge un modèle entraîné depuis data/processed/<name>."""
    device = device or _device()
    _, maps = load_all()
    ckpt = torch.load(PROCESSED / name, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TrajectoryJEPA(cfg["n_tracks"], cfg["n_artists"], maps.track2artist,
                           d_model=cfg["d_model"], nhead=cfg["nhead"],
                           session_layers=cfg.get("session_layers", 2),
                           temporal_layers=cfg.get("temporal_layers", 2),
                           pred_hidden=cfg.get("pred_hidden", 256),
                           max_session_len=cfg.get("max_session_len", SESSION_CAP),
                           max_sessions=cfg.get("max_sessions", WINDOW), ema=cfg["ema"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    run()
