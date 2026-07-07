"""Entraînement du set-transformer de prédiction de notes masquées (résidu).

Cible = résidu `note − μ − b_i − b_u`. Perte = MSE sur les positions masquées. On
surveille par epoch la RMSE de VALIDATION reconstruite `clamp(μ+b_i+b_u+résidu_hat)`,
affichée à côté du MUR baseline (μ+b_i+b_u seul) — le juge direct.

Split 3 voies sans fuite : test (20 %, juge final) et val (10 % du reste, suivi) sont
tenus hors du pool d'entraînement ET hors de l'estimation de μ/b_i.

Calqué sur `train_jepa.py` (device, AMP CUDA-only, warmup/cosine, checkpoint/epoch).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_prep import PROCESSED, load_all
from .data_ratings import (RATING_MIN, RATING_MAX, split_user_items, carve_val,
                           union_masks, global_biases,
                           RatingTrainDataset, RatingEvalDataset, collate)
from .rating_model import SetRatingModel
from .train_jepa import _lr_factor


@torch.no_grad()
def _val_rmse(model, loader, device, max_batches: int | None = None):
    """RMSE reconstruite (modèle) et RMSE du mur (baseline μ+b_i+b_u) sur le held-out."""
    model.eval()
    se_model = se_base = mae_model = n = 0.0
    for bi, batch in enumerate(loader):
        items = batch["items"].to(device)
        levels = batch["levels"].to(device)
        bias = batch["bias"].to(device)
        kpm = batch["key_padding_mask"].to(device)
        tmask = batch["target_mask"].to(device)
        base = batch["target_baseline"].to(device)
        true = batch["target_true"].to(device)

        resid, _ = model(items, levels, bias, kpm)
        pred = torch.clamp(base + resid, RATING_MIN, RATING_MAX)
        m = tmask
        se_model += float(((pred[m] - true[m]) ** 2).sum())
        mae_model += float((pred[m] - true[m]).abs().sum())
        se_base += float(((torch.clamp(base[m], RATING_MIN, RATING_MAX) - true[m]) ** 2).sum())
        n += int(m.sum())
        if max_batches and bi + 1 >= max_batches:
            break
    model.train()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return (np.sqrt(se_model / n), np.sqrt(se_base / n), mae_model / n)


def run(epochs: int = 15, batch_size: int = 128, d_model: int = 128, nhead: int = 4,
        layers: int = 4, head_hidden: int = 256, mask_frac: float = 0.2,
        lr: float = 1e-3, lam_i: float = 5.0, subset: int | None = None,
        seed: int = 0, num_workers: int = 2, ckpt_name: str = "ratings.pt",
        device: str | None = None):
    # MPS : la self-attention masquée (padding) NaN + très lente pour ce modèle.
    # Cible réelle = CUDA (Colab, AMP). Local : forcer CPU (correct, cf. smoke).
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    print(f"device={device} | epochs={epochs} | batch={batch_size} | workers={num_workers}", flush=True)
    print(f"archi : d_model={d_model} nhead={nhead} layers={layers} head_hidden={head_hidden} "
          f"mask_frac={mask_frac}", flush=True)

    sequences, genome, maps = load_all()
    if subset:
        sequences = sequences.iloc[:subset].reset_index(drop=True)
        print(f"SUBSET actif : {len(sequences):,} users", flush=True)

    # Split 3 voies + biais globaux estimés sur le TRAIN seulement (val+test exclus)
    test_masks = split_user_items(sequences, frac=0.2, seed=seed)
    val_masks = carve_val(sequences, test_masks, frac=0.1, seed=seed + 7)
    heldout = union_masks(test_masks, val_masks)
    mu, b_i = global_biases(sequences, heldout, maps.n_items, lam_i=lam_i)
    print(f"μ={mu:.4f} | b_i sur {int((b_i != 0).sum()):,} films | "
          f"held-out (val+test) exclu des biais", flush=True)

    train_ds = RatingTrainDataset(sequences, heldout, mu, b_i, mask_frac=mask_frac)
    val_ds = RatingEvalDataset(sequences, val_masks, mu, b_i)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              drop_last=True, num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=num_workers > 0)
    print(f"train users={len(train_ds):,} | val users={len(val_ds):,} | "
          f"steps/epoch={len(train_loader):,}", flush=True)

    model = SetRatingModel(maps.n_items, genome, d_model=d_model, nhead=nhead,
                           layers=layers, head_hidden=head_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params entraînables : {n_params/1e6:.2f} M", flush=True)
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
        agg_loss = torch.zeros((), device=device)
        n_agg = 0
        for batch in train_loader:
            items = batch["items"].to(device)
            levels = batch["levels"].to(device)
            bias = batch["bias"].to(device)
            kpm = batch["key_padding_mask"].to(device)
            tmask = batch["target_mask"].to(device)
            tresid = batch["target_resid"].to(device)

            for g in opt.param_groups:
                g["lr"] = lr * _lr_factor(step, total_steps)
            opt.zero_grad()
            with torch.autocast(device_type=amp_dev, dtype=torch.float16, enabled=use_amp):
                resid, _ = model(items, levels, bias, kpm)
                loss = torch.nn.functional.mse_loss(resid[tmask], tresid[tmask])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            agg_loss += loss.detach()
            n_agg += 1
            step += 1
            if n_agg % 100 == 0:
                print(f"  epoch {epoch+1:2d} | step {n_agg:4d}/{steps_per_epoch} | "
                      f"loss {float(agg_loss)/n_agg:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)

        train_mse = float(agg_loss) / max(1, n_agg)
        val_rmse, base_rmse, val_mae = _val_rmse(model, val_loader, device)
        dt = time.time() - t0
        rec = {"epoch": epoch + 1, "train_mse": train_mse, "val_rmse": val_rmse,
               "val_mae": val_mae, "baseline_rmse": base_rmse, "elapsed_s": round(dt, 1)}
        history.append(rec)
        gain = base_rmse - val_rmse
        print(f"epoch {epoch+1:2d}/{epochs} | train_mse {train_mse:.4f} | "
              f"val RMSE {val_rmse:.4f} vs mur {base_rmse:.4f} "
              f"({'+' if gain>=0 else ''}{gain:.4f}) | {dt/60:.1f} min", flush=True)

        PROCESSED.mkdir(parents=True, exist_ok=True)
        ckpt = {"state_dict": model.state_dict(),
                "config": {"n_items": maps.n_items, "d_model": d_model, "nhead": nhead,
                           "layers": layers, "head_hidden": head_hidden},
                "biases": {"mu": mu, "b_i": b_i}, "epoch": epoch + 1,
                "split": {"frac_test": 0.2, "frac_val": 0.1, "seed": seed}}
        torch.save(ckpt, PROCESSED / ckpt_name)
        with open(PROCESSED / f"history_{ckpt_name.replace('.pt', '')}.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nCheckpoint -> {PROCESSED/ckpt_name}", flush=True)
    return history


def load_model(device: str | None = None, name: str = "ratings.pt"):
    """Recharge (modèle, mu, b_i) depuis `data/processed/<name>`."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, genome, _ = load_all()
    ckpt = torch.load(PROCESSED / name, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SetRatingModel(cfg["n_items"], genome, d_model=cfg["d_model"], nhead=cfg["nhead"],
                           layers=cfg["layers"], head_hidden=cfg["head_hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    b = ckpt["biases"]
    return model, b["mu"], b["b_i"]


if __name__ == "__main__":
    run()
