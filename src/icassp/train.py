import argparse
import os
import shutil
from datetime import date
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from icassp.dataset import RP_VADataset
from icassp.ddpm import DDPM, DDPMLightVA, ResUnet
from icassp.gan import Discriminator, GANlight, Generator
from icassp.utils import (
    compute_metrics,
    get_save_path_create_folder,
    load_scheduler_from_config,
    load_config,
    selectRP,
    set_rp_length,
    train_val_split,
)


def _build_dataloaders(dataset, batch_size, num_workers, generalize, seed, rng):
    train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx, test_mmsis = train_val_split(
        dataset,
        generalize=generalize,
        val_size=0.10,
        seed=seed,
        rng=rng,
    )

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_dataset, train_idx, val_idx, test_idx, test_mmsis


def _resolve_trainer_args(config, logger, callbacks):
    use_gpu = torch.cuda.is_available()
    devices = torch.cuda.device_count() if use_gpu else 1
    strategy = "ddp" if devices and devices > 1 else None
    precision = "16-mixed" if use_gpu else 32
    val_interval = max(1, int(config["val_every_n_data"] // config["batch_size"]))

    trainer_kwargs = dict(
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        max_epochs=config["epochs"],
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        precision=precision,
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
    )
    if strategy:
        trainer_kwargs["strategy"] = strategy

    if config["model"] == "ddpm":
        trainer_kwargs["gradient_clip_algorithm"] = "norm"
        trainer_kwargs["gradient_clip_val"] = 0.5

    return trainer_kwargs


def build_model(config, dataset, val_idx, test_idx, save_path, minmax):
    if config["model"] == "ddpm":
        scheduler = load_scheduler_from_config(config)
        unet = ResUnet(config=config)
        ddpm = DDPM(unet, config["num_timesteps"], var_scheduler=scheduler)
        model = DDPMLightVA(ddpm, config, dataset, [val_idx, test_idx], save_path, minmax)
    elif config["model"] == "gan":
        model = GANlight(Generator, Discriminator, config, dataset, [val_idx, test_idx], save_path, minmax)
        ddpm = None
    else:
        raise ValueError(f"Unknown model type {config['model']}")
    return model, ddpm


def parse_args():
    default_workers = max((os.cpu_count() or 2) - 1, 1)
    parser = argparse.ArgumentParser(description="Train DDPM or GAN models on HRRP data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gan_scalars_serloss.yaml",
        help="Path to the YAML configuration file (relative to repo root).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ship_hrrp.pt",
        help="Path to the dataset file (.pt demo set or a CSV with HRRP columns).",
    )
    parser.add_argument("--iteration", type=int, default=1, help="Run identifier added to the log folder name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help="Number of workers for dataloaders.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Disable test metrics computation after training (useful when no test split is available).",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Lightning fast dev run (1 train/val/test batch) to sanity-check the pipeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path.resolve()}")

    config = load_config(config_path)
    set_rp_length(config.get("rp_length", len(selectRP)))
    batch_size = config["batch_size"]

    dataset = RP_VADataset(config, path_rp=args.data)
    min_rp, max_rp = dataset.min_rp, dataset.max_rp

    train_loader, val_loader, test_dataset, train_idx, val_idx, test_idx, test_mmsis = _build_dataloaders(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        generalize=config.get("generalize", False),
        seed=args.seed,
        rng=config.get("valrange"),
    )

    monitor_metric = "val_loss" if config["model"] == "ddpm" else "val_mse"
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="results/lightning_logs/{}".format(date.today()),
        name=config["figure_path"].split(" ")[-1] + "_" + str(args.iteration),
    )
    early_stop = EarlyStopping(monitor=monitor_metric, patience=1000, mode="min")

    save_path = get_save_path_create_folder(config, args.seed)
    checkpoint_cb = ModelCheckpoint(
        dirpath=save_path,
        filename=config["model_name"],
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
    )

    shutil.copyfile(config_path, Path(save_path) / "config.yaml")

    model, ddpm = build_model(config, dataset, val_idx, test_idx, save_path, (min_rp, max_rp))
    callbacks = [checkpoint_cb, early_stop]

    trainer_args = _resolve_trainer_args(config, tb_logger, callbacks)
    if args.fast_dev_run:
        trainer_args["fast_dev_run"] = True
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Reload best checkpoint for evaluation
    best_ckpt = checkpoint_cb.best_model_path
    if not best_ckpt:
        print("No checkpoint was saved; skipping reload.")
        return

    if config["model"] == "ddpm":
        model = DDPMLightVA.load_from_checkpoint(
            best_ckpt,
            ddpm=ddpm,
            config=config,
            dataset=dataset,
            validation_indices=[val_idx, test_idx],
            save_path=save_path,
            minmax=(min_rp, max_rp),
        )
    else:
        model = GANlight.load_from_checkpoint(
            best_ckpt,
            gen=Generator,
            disc=Discriminator,
            config=config,
            dataset=dataset,
            validation_indices=[val_idx, test_idx],
            save_path=save_path,
            minmax=(min_rp, max_rp),
        )
    model.eval()

    if not args.skip_eval and test_idx:
        metrics = compute_metrics(model, dataset, test_idx, min_rp, max_rp, config["model"])
        print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
