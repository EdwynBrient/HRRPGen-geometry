from __future__ import annotations

import argparse
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset

try:
    from .ddpm import DDPM, DDPMlight1D2D, ResUnet
    from .gan import Discriminator, GANlight, Generator
    from .utils import get_save_path_create_folder, load_config, load_scheduler_from_config, train_val_split
except ImportError:  # pragma: no cover
    from ddpm import DDPM, DDPMlight1D2D, ResUnet
    from gan import Discriminator, GANlight, Generator
    from utils import get_save_path_create_folder, load_config, load_scheduler_from_config, train_val_split


class PTScalarsDataset(Dataset):
    """Tiny dataset adapter for `.pt` HRRP payloads used in repo quick tests."""

    def __init__(self, pt_path: Path):
        super().__init__()
        payload = torch.load(pt_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Le fichier .pt doit contenir un dict (keys: hrrps/scals).")
        if "hrrps" not in payload:
            raise KeyError("Key `hrrps` absente du .pt")

        hrrps = torch.as_tensor(payload["hrrps"], dtype=torch.float32)
        if hrrps.ndim != 2:
            raise ValueError(f"`hrrps` doit être [N, L], reçu {tuple(hrrps.shape)}")
        if hrrps.shape[1] != 200:
            raise ValueError(f"`hrrps` doit avoir longueur 200, reçu {hrrps.shape[1]}")

        if "scals" in payload:
            scals = torch.as_tensor(payload["scals"], dtype=torch.float32)
            if scals.ndim != 2 or scals.shape[1] != 3:
                raise ValueError(f"`scals` doit être [N,3], reçu {tuple(scals.shape)}")
            angles = scals[:, 0]
            lengths = scals[:, 1]
            widths = scals[:, 2]
        else:
            if "aspect_angles" not in payload or "ship_dims" not in payload:
                raise KeyError("Fournir `scals` ou (`aspect_angles` et `ship_dims`) dans le .pt")
            angles = torch.as_tensor(payload["aspect_angles"], dtype=torch.float32).reshape(-1)
            dims = torch.as_tensor(payload["ship_dims"], dtype=torch.float32)
            if dims.ndim != 2 or dims.shape[1] != 2:
                raise ValueError(f"`ship_dims` doit être [N,2], reçu {tuple(dims.shape)}")
            lengths = dims[:, 0]
            widths = dims[:, 1]

        n = hrrps.shape[0]
        if not (len(angles) == len(lengths) == len(widths) == n):
            raise ValueError("Tailles incohérentes entre `hrrps` et les scalaires")

        self.min_rp = 0.0
        self.max_rp = float(torch.max(hrrps).item())
        den = self.max_rp - self.min_rp
        if den <= 1e-12:
            den = 1.0
        self.hrrps = ((hrrps - self.min_rp) / den) * 2.0 - 1.0

        len_max = float(torch.max(lengths).item()) if lengths.numel() else 1.0
        wid_max = float(torch.max(widths).item()) if widths.numel() else 1.0
        if len_max <= 1e-12:
            len_max = 1.0
        if wid_max <= 1e-12:
            wid_max = 1.0

        self.viewing_angles = angles
        self.lengths = lengths / len_max
        self.widths = widths / wid_max
        self.mmsis = np.arange(n, dtype=np.int64)
        self.old_va = pd.Series(self.viewing_angles.numpy().copy())
        va_max = float(self.old_va.max()) if len(self.old_va) else 1.0
        if va_max <= 1e-12:
            va_max = 1.0

        df = pd.DataFrame({str(i): self.hrrps[:, i].numpy() for i in range(200)})
        df["mmsi"] = self.mmsis
        df["length"] = self.lengths.numpy()
        df["width"] = self.widths.numpy()
        df["viewing_angle"] = self.viewing_angles.numpy() / va_max
        df["ship_type_mode_int"] = 0
        self.df = df

    def __len__(self) -> int:
        return self.hrrps.shape[0]

    def __getitem__(self, idx: int):
        hrrp = self.hrrps[idx]
        va = self.viewing_angles[idx:idx + 1]
        length = self.lengths[idx:idx + 1]
        width = self.widths[idx:idx + 1]
        vars_ = torch.cat([hrrp.unsqueeze(0), va.unsqueeze(0), length.unsqueeze(0), width.unsqueeze(0)], dim=1)
        return vars_, torch.tensor([idx], dtype=torch.uint32)


def _resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p.resolve()
    return (project_root / value).resolve()


def _prepare_config_defaults(config: dict) -> dict:
    config["vt_frac"] = config.get("vt_frac", 0.2)
    config["lambda_reg"] = config.get("lambda_reg", 0.01)
    config["lambda_cos"] = config.get("lambda_cos", 0.01)
    config["val_metrics_max_samples"] = config.get("val_metrics_max_samples", 1)
    config["inf_mode"] = config.get("inf_mode", "epoch")
    config["inf_every_n_step"] = config.get("inf_every_n_step", 10_000)
    config["inf_every_n_epoch"] = config.get("inf_every_n_epoch", 1)
    config["tvmf"] = config.get("tvmf", 0.0)

    cond = config.get("conditionned", {})
    cond["cond_op"] = cond.get("cond_op", "add")
    config["conditionned"] = cond

    fidelity = config.get("Fidelity", {})
    fidelity["bool"] = fidelity.get("bool", 0)
    fidelity["lambda_l1"] = fidelity.get("lambda_l1", 0.0)
    fidelity["lambda_grad"] = fidelity.get("lambda_grad", 0.0)
    fidelity["lambda_fft"] = fidelity.get("lambda_fft", 0.0)
    fidelity["lambda_feat"] = fidelity.get("lambda_feat", 0.0)
    config["Fidelity"] = fidelity
    return config


def main():
    parser = argparse.ArgumentParser(description="Mini training entrypoint for HRRP repo")
    parser.add_argument("--config", type=str, default="configs/gan_scalars.yaml", help="YAML config path")
    parser.add_argument("--data", type=str, default="data/ship_hrrp.pt", help="Input .pt dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true", help="Conservé pour compat CLI")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 1 batch train/val")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = _resolve_path(project_root, args.config)
    data_path = _resolve_path(project_root, args.data)

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data introuvable: {data_path}")
    if data_path.suffix.lower() != ".pt":
        raise ValueError("Ce mode de test attend un dataset `.pt`.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = _prepare_config_defaults(load_config(str(config_path)))
    config.setdefault("conditionned", {})
    config["conditionned"]["type"] = "scalars"

    dataset = PTScalarsDataset(data_path)
    min_rp, max_rp = dataset.min_rp, dataset.max_rp

    train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, _ = train_val_split(
        dataset,
        generalize=config.get("generalize", 0),
        val_size=0.10,
        seed=args.seed,
        rng=config.get("valrange", None),
    )

    bs = int(config.get("batch_size", 32))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(bs, max(1, len(train_ds))),
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=max(1, min(bs, max(1, len(val_ds)))),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_type = str(config.get("model", "gan")).lower()
    if model_type == "ddpm":
        monitor = "val_loss"
    elif model_type in {"gan", "gen"}:
        monitor = "val_mse"
    else:
        raise ValueError(f"Model `{model_type}` non supporté dans ce mode mini-test.")

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(project_root / f"results/lightning_logs/{date.today()}"),
        name=f"{config.get('model_name', model_type)}_{args.iteration}",
    )
    early_stop_callback = EarlyStopping(monitor=monitor, patience=1000, mode="min")

    save_path_env_name = get_save_path_create_folder(config, args.seed)
    model_checkpoint = ModelCheckpoint(
        dirpath=save_path_env_name,
        filename=config.get("model_name", model_type),
        monitor=monitor,
        save_top_k=1,
        mode="min",
    )
    shutil.copyfile(config_path, os.path.join(save_path_env_name, "config.yaml"))

    if model_type == "ddpm":
        scheduler = load_scheduler_from_config(config)
        unet = ResUnet(config=config)
        ddpm = DDPM(unet, int(config.get("num_timesteps", 1000)), var_scheduler=scheduler)
        model = DDPMlight1D2D(ddpm, config, dataset, [val_idx, test_idx], save_path_env_name, (min_rp, max_rp))
    else:
        model = GANlight(Generator, Discriminator, config, dataset, [val_idx, test_idx], save_path_env_name, (min_rp, max_rp))

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = config.get("precision", "16-mixed")
    else:
        accelerator = "cpu"
        devices = 1
        precision = "32-true"

    val_interval_steps = max(1, int(config.get("val_every_n_data", bs)) // max(1, bs))
    val_interval_steps = min(val_interval_steps, max(1, len(train_loader)))

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(args.max_epochs if args.max_epochs is not None else config.get("epochs", 1)),
        val_check_interval=val_interval_steps,
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        precision=precision,
        num_sanity_val_steps=0,
        logger=tb_logger,
        detect_anomaly=False,
        callbacks=[model_checkpoint, early_stop_callback],
        fast_dev_run=args.fast_dev_run,
    )

    print(f"[mini-train] dataset size={len(dataset)} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[mini-train] model={model_type} config={config_path} data={data_path}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("[mini-train] training terminé.")


if __name__ == "__main__":
    main()
