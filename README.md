# HRRP Missing Scenarios

DDPM / GAN for High-Resolution-Range-Profile generation, in proceedings at EUSIPCO26.
Executable code lives in `src/ship_hrrp_gen`, configs in `configs/`.
<figure>
  <img src="assets/ship_hrrp.png" alt="Generated samples overview" />
  <figcaption>Figure 1 — Instance of generated HRRP data from `data/ship_hrrp.pt` </figcaption>
</figure>

# Top contribution

## Fundamental conditions

We show that the ship's dimensions and its aspect angle at acquisition are mandatory conditions for generating a ship-specific HRRP. These conditions are interdependent, as shown in the table below.

<figure>
  <img src="assets/comparison_lenrp.png" alt="LRP LOSP" />
  <figcaption>Table 1 — Generation metrics for different models and conditioning types. The best scores for each model are in bold. </figcaption>
</figure>

## Generated HRRP follow TLOP

Although HRRP data are inherently noisy and difficult to interpret, the theoretical target length observed in a radar HRRP follows a well-defined geometric relationship with the target’s aspect angle. This relationship is described by the Theoretical Length of Object Projection (TLOP)_ model:

TLOP(L, W, asp) = |L · cos(asp)| + |W · sin(asp)|

where _L_ and _W_ denote the target's true **length** and **width**, and _asp_ is target's **aspect angle** at acquisition time.

Using a robust detection of the target’s occupied range bins, the visual target's length called _Length on Range Profile_ (LRP) can be estimated directly from HRRP data. As shown in the accompanying figure, these measured lengths exhibit a clear correlation with the theoretical _TLOP_ curves, confirming that the physical projection geometry is preserved in real radar measurements.

The same analysis applied to the **generated HRRP data** shows that the synthesized signals exhibit LOSP-consistent trends and successfully **fill missing aspect-angle scenarios** at a coarse scale. This demonstrates that the generation process preserves the underlying physical and geometric constraints of radar line-of-sight projections, beyond simple signal-level realism.

<figure>
  <img src="assets/comparison_lrp.png" alt="LRP LOSP" />
  <figcaption>Figure 2 — Correlation between visual <i>Length on Range Profile</i> (LRP) and <i>Theoretical Length of Object Projection</i> (TLOP) for measured and generated data. </figcaption>
</figure>

## Quick requirements
- Python ≥ 3.9
- PyTorch (CPU or CUDA, matching your GPU if available)
- The generated demo set (`data/ship_hrrp.pt`).

## Installation
```bash
cd github_repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # installs the ship_hrrp_gen package from src/
```

## Run a training
```bash
python -m ship_hrrp_gen.train \
  --config configs/gan_scalars_serloss.yaml \
  --data data/ship_hrrp.pt \
  --seed 42 \
  --num-workers 0
```
Useful flags:
- `--num-workers`: dataloader workers (set 0 on small CPUs).
- `--skip-eval`: skip test metrics if no test split.
- `--fast-dev-run`: quick pipeline sanity-check (1 train/val/test batch).

## Quick sanity check
```bash
python -m ship_hrrp_gen.train \
  --config configs/gan_scalars_serloss.yaml \
  --data data/ship_hrrp.pt \
  --seed 0 \
  --num-workers 0 \
  --skip-eval \
  --fast-dev-run
```

Artifacts (checkpoints, figures, TensorBoard logs) are written under `results/` following the `figure_path` in the config.

## Layout
- `src/ship_hrrp_gen/`: models (DDPM, GAN), dataset, utils, training script (`train.py`).
- `configs/`: all YAML configs.
- `requirements.txt`: Python deps.
- `.gitignore`: ignores caches, venvs, and training outputs.

## Notes
- Intended for a quick demo run on the 128 generated samples; no multi-GPU setup required.
- Final metrics rely on `compute_metrics` in `ship_hrrp_gen.utils`. If there is no test split (`test_idx` empty), use `--skip-eval`.

## Citation 

```
@inproceedings{brient2026eusipco,
  author    = {Edwyn Brient and Santiago Velasco{-}Forero and Rami Kassab},
  title     = {{Conditional Generative Models for High-Resolution Range Profile: Capturing Global Trends in a Large-Scale Maritime Dataset}},
  booktitle = {Proc. European Signal Processing Conference (EUSIPCO)},
  year      = {2026},
  note      = {to appear}
}
```
