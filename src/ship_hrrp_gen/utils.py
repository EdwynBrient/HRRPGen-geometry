import numpy as np
import pandas as pd
import cv2
import skimage
import torch
from torch.utils.data import random_split
import os
import yaml
from datetime import date
try:
    from .schedulers import *
except ImportError:  # pragma: no cover
    from schedulers import *
from torchvision.transforms import functional as Func
try:
    from ignite.metrics import MaximumMeanDiscrepancy
except Exception:  # pragma: no cover - optional dependency
    MaximumMeanDiscrepancy = None
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import sys
import math
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

selectRP = [str(i) for i in range(200)]

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


# class MMDLoss(nn.Module):

#     def __init__(self, kernel=RBF()):
#         super().__init__()
#         self.kernel = kernel

#     def forward(self, X, Y):
#         K = self.kernel(torch.vstack([X, Y]))

#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY

def plot_test_argmins(HRRP, generated, values, path, i):
    """In the same figure, plot one column with the true HRRP, and one column with the generated HRRP
    HRRP : (N, 200) 
    generated : (N, 200)
    """
    # Reshape inputs to 3D format expected by plot_HRRP (rows, cols, 200)
    HRRP = HRRP.reshape(-1, 1, 200)
    generated = generated.reshape(-1, 1, 200)
    
    # Stack true and generated side by side
    combined = np.concatenate([HRRP, generated], axis=1)
    
    # Create labels for the columns
    labels = np.array([["True HRRP {}".format(values[i]), "Generated HRRP"] for i in range(HRRP.shape[0])])
    if combined.shape[0]!=1:    
        # Use plot_HRRP with the combined data
        plot_HRRP(combined, "Worst True vs Generated HRRPs", labels, path+"worst_{}".format(i), save=True)
        plt.close("all")

def plot_generated_true(HRRP, generated, scal, save_path, min_rp, max_rp, mode, epoch):
    HRRP = HRRP.detach().cpu().numpy()
    generated = np.array(generated)
    save_path = os.path.join(save_path, "figures")
    os.makedirs(save_path, exist_ok=True)
    little_title = [", ".join(str(np.round(val, 2)) for val in scal) for scal in scal]
    little_title = np.array(little_title).reshape(-1, 6)
    HRRP, generated = unnormalize_hrrp(HRRP, min_rp, max_rp), unnormalize_hrrp(generated, min_rp, max_rp)
    HRRP, generated = np.reshape(HRRP, (-1, 6, 200)), np.reshape(generated, (-1, 6, 200))
    plot_HRRP(HRRP, "True HRRP", little_title, save_path + "/" + "true_hrrps_" + mode + str(epoch) + ".png", True)
    plot_HRRP(generated, "Final Generated HRRP", little_title, save_path + "/" + "final_gen_hrrps_" + mode + str(epoch) + ".png", True)
    HRRP, generated = np.reshape(HRRP, (-1, 200)), np.reshape(generated, (-1, 200))
    plt.close("all")
    return HRRP, generated

def prepare_data(df, i, test_idx, vamax, bs):
    if bs*(i+1) <= len(test_idx):
        idx = test_idx[bs*i:bs*(i+1)]
    else:
        idx = test_idx[bs*i:]
    data = df.iloc[idx]
    vars = []
    hrrps = df[selectRP].to_numpy()
    for i in range(len(data)):
        hrrp = torch.Tensor(hrrps[i]).unsqueeze(0)
        va = torch.Tensor([data.iloc[i]["viewing_angle"]*vamax]).unsqueeze(0)
        length = torch.Tensor([data.iloc[i]["length"]]).unsqueeze(0)
        width = torch.Tensor([data.iloc[i]["width"]]).unsqueeze(0)
        vars.append(torch.cat([hrrp, va, length, width], dim=1))
    return vars, idx

def compute_metrics(model, dataset, test_idx, min_rp, max_rp, model_type, path=None):
    """Evaluate model on a subset of indices.

    The previous version always concatenated 3 scalars (va, length, width) and
    attempted to build masks for any non-"scalars" model. This broke for
    "dims"-only GANs: the scalar embedding expected 2 scalars (length/width),
    but 3 were provided, leading to a Linear shape mismatch. Masks are only
    needed when the model actually uses them.
    """
    bs = 128
    gen, rp = [], []
    tpsnr, tcosf, tmsef = [], [], []
    lpf, rcs = [], []
    use_masks = model.type not in ("scalars", "dims")

    with torch.no_grad():
        for i in tqdm(range(0, len(test_idx) // bs), file=sys.stdout):
            vars, idx = prepare_data(dataset.df, i, test_idx, dataset.old_va.max(), bs)
            vars = torch.stack(vars, dim=0).to("cuda")
            RP = vars[:, :, :200].float().cpu()

            # Build scalars exactly as the trained GAN expects
            if hasattr(model, "_extract_scalars"):
                scal = model._extract_scalars(vars).to("cuda")
            else:
                # fallback (DDPM or other models needing full scalar triplet)
                scal = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float().to("cuda")

            # Only build masks when the architecture really needs them
            if use_masks:
                mmsi_batch = dataset.df.mmsi.iloc[idx].values
                va_batch = dataset.df.viewing_angle.iloc[idx].values * dataset.old_va.max()
                mask1_list, mask2_list = [], []
                for mmsi_val, va_val in zip(mmsi_batch, va_batch):
                    m1, m2 = dataset.select_masks(mmsi_val, va_val)
                    mask1_list.append(m1)
                    mask2_list.append(m2)
                mask1 = torch.stack(mask1_list).float().to("cuda")
                mask2 = torch.stack(mask2_list).float().to("cuda")
                masks = [mask1, mask2]

            if model_type == "ddpm":
                generated, RP = model([vars]) if not use_masks else model([masks, vars])
                generated = torch.Tensor(np.array(generated))
            else:
                z = torch.randn(vars.shape[0], 1, 25).to("cuda")
                if use_masks:
                    generated = model(z, [masks, scal])
                else:
                    generated = model(z, scal)

            generated = unnormalize_hrrp(generated.cpu(), min_rp, max_rp)
            RP = unnormalize_hrrp(RP, min_rp, max_rp)
            idx = torch.Tensor(idx).to(torch.int64)
            metrics = model.compute_loss("test", idx, generated)

            lpf.append(metrics[3])
            rcs.append(metrics[4])
            tpsnr.append(metrics[0])
            tcosf.append(metrics[1])
            tmsef.append(metrics[2])
            gen.append(generated)
            rp.append(RP)

    psnr = torch.cat(tpsnr, dim=0)
    tcosf = torch.cat(tcosf, dim=0)
    tmsef = torch.cat(tmsef, dim=0)
    lpf = torch.cat(lpf, dim=0)
    rcs = torch.cat(rcs, dim=0)
    return psnr, tcosf, tmsef, lpf, rcs

def cache_embeddings(emb_dict, emb_base_path="./embeddings"):
    os.makedirs(emb_base_path, exist_ok=True)
    for mmsi in emb_dict.keys():
        os.makedirs(os.path.join(emb_base_path, str(mmsi)), exist_ok=True)
        for png in emb_dict[mmsi].keys():
            numpy_file = emb_dict[mmsi][png].detach().cpu().numpy()
            np.save(os.path.join(emb_base_path, str(mmsi), png)+".npy", numpy_file)

def load_embeddings(emb_base_path="./embeddings"):
    emb_dict = {}
    for mmsi in os.listdir(emb_base_path):
        emb_dict[mmsi] = {}
        for png in os.listdir(os.path.join(emb_base_path, mmsi)):
            emb_dict[mmsi][png] = torch.from_numpy(np.load(os.path.join(emb_base_path, mmsi, png)))
    return emb_dict

def check_mmsi_in_folder(mmsi_list, emb_path):
    if not os.path.exists(emb_path):
        return False
    for mmsi in mmsi_list:
        if mmsi not in os.listdir(emb_path):
            print(f"Missing mmsi {mmsi} in embeddings folder, recomputing embeddings.")
            return False
        
    return True

def update_grad_lists(model, gradients_norm, gradients_max_val):
    gradients_norm = []
    gradients_max_val = []
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
        max_grad = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
        max_grad = torch.max(torch.stack([torch.max(p.grad.detach()) for p in parameters])).item()
    gradients_norm.append(total_norm)
    gradients_max_val.append(max_grad)
    return gradients_norm, gradients_max_val

def pad_to_square(image, fill=255.):
    width, height = image.size
    max_side = max(width, height)
    padding = ((max_side - width)//2, (max_side - height)//2)  # (left-right, top-bottom)
    return Func.pad(image, padding, fill=fill) 

def train_val_split(dataset, generalize=False, val_size=0.2, seed=8, rng=None, test=True):
    """
    Split rapide:
    - generalize=False: utilise random_split classique (réproducible).
    - generalize=True & rng is None: tire sur MMSI *uniques* (val/test par groupes MMSI).
    - generalize=True & rng float in [0,1]: prend une fenêtre contiguë d'indices pour val, reste en train.
    Retourne: (train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, test_mmsis)
    """
    n = len(dataset)
    all_idx = np.arange(n, dtype=np.int64)

    if not generalize:
        g = torch.Generator().manual_seed(seed)
        n_val = int(round(n * val_size))
        train_len = n - n_val
        train_ds, val_ds = random_split(dataset, [train_len, n_val], generator=g)
        # Pas de test split ici (comme ta version)
        empty = []
        return train_ds, val_ds, torch.utils.data.Subset(dataset, empty), \
               list(range(train_len)), list(range(train_len, n)), empty, []

    # ----- generalize=True -----
    rng_np = np.random.default_rng(seed)

    if rng is None or rng == "None":
        # Tirage par MMSI uniques, version vectorisée
        mmsi_col = dataset.df["mmsi"].to_numpy()  # (N,)
        unique_mmsis = np.unique(mmsi_col)
        nb_mmsis = int(len(unique_mmsis) * val_size)

        if nb_mmsis == 0:
            # fallback: rien en val, tout en train
            train_idx = all_idx
            val_idx = np.array([], dtype=np.int64)
            test_idx = np.array([], dtype=np.int64)
            test_mmsis = np.array([], dtype=unique_mmsis.dtype)
        else:
            val_mmsis = rng_np.choice(unique_mmsis, nb_mmsis, replace=False)
            test_mmsis = np.array([], dtype=unique_mmsis.dtype)
            if test:
                # moitié des MMSI de val -> test
                nb_test = max(nb_mmsis // 2, 1) if nb_mmsis > 1 else 0
                if nb_test > 0:
                    test_mmsis = rng_np.choice(val_mmsis, nb_test, replace=False)
                    # MMSI de validation = val_mmsis \ test_mmsis
                    val_mmsis = np.setdiff1d(val_mmsis, test_mmsis, assume_unique=False)

            mask_val  = np.isin(mmsi_col, val_mmsis)
            mask_test = np.isin(mmsi_col, test_mmsis) if test_mmsis.size > 0 else np.zeros(n, dtype=bool)
            mask_train = ~(mask_val | mask_test)

            val_idx   = all_idx[mask_val]
            test_idx  = all_idx[mask_test]
            train_idx = all_idx[mask_train]

        # Construire les Subset
        train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
        test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())

        return train_ds, val_ds, test_ds, \
               train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), test_mmsis.tolist()

    else:
        # rng numérique: fenêtre contiguë d'indices (vectorisée)
        r = float(rng)
        r = 0.0 if r < 0 else (1.0 if r > 1.0 else r)
        if r == 1.0:
            r = r - val_size  # pour avoir au moins un échantillon en val

        start = int(r * n)
        n_val = int(val_size * n)
        stop  = min(start + n_val, n)
        val_idx = np.arange(start, stop, dtype=np.int64)

        if test and val_idx.size > 0:
            # moitié des indices de val pour test (tirage stable)
            test_idx = rng_np.choice(val_idx, val_idx.size // 2, replace=False)
            # retirer test de val
            keep_mask = np.ones(val_idx.shape[0], dtype=bool)
            # marquer comme faux les positions qui appartiennent à test_idx
            # méthode rapide: créer un set des valeurs test pour lookup O(1)
            test_set = set(test_idx.tolist())
            for j, v in enumerate(val_idx):
                if v in test_set:
                    keep_mask[j] = False
            val_idx = val_idx[keep_mask]
        else:
            test_idx = np.array([], dtype=np.int64)

        keep = np.ones(n, dtype=bool)
        keep[val_idx] = False
        keep[test_idx] = False
        train_idx = all_idx[keep]

        train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
        test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())
        return train_ds, val_ds, test_ds, \
               train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

def top_loss(df, idx, generated, minrp, maxrp, tol_angle=0.01):
    """Compute the min MSE loss of all HRRPs of the same ship at tol_angle (rad)"""
    # Extract relevant data
    mmsi = df.iloc[idx].mmsi
    df_mmsi = df[df.mmsi == mmsi].copy()
    df_mmsi[selectRP] = unnormalize_hrrp(df_mmsi[selectRP].to_numpy(), minrp, maxrp)
    va = df.iloc[int(idx)].viewing_angle * 6.28
    df_va_filtered = df_mmsi.viewing_angle * 6.28  # Keep original indices

    # Define range bounds
    lower_bound = (va - tol_angle) % (2 * np.pi)
    upper_bound = (va + tol_angle) % (2 * np.pi)

    # Handle cases where range wraps around 0 or 2π
    if lower_bound < upper_bound:
        mask_va = (df_va_filtered > lower_bound) & (df_va_filtered < upper_bound)
    else:
        # Wrap-around case: angles are in two separate intervals
        mask_va = (df_va_filtered > lower_bound) | (df_va_filtered < upper_bound)

    df_around = df_mmsi[mask_va]
    rp_around = df_around[selectRP].to_numpy()

    generated = np.expand_dims(generated, axis=0)
    
    # Compute MFN calculate cosine & MSE
    rp_around = torch.Tensor(rp_around)
    generated = torch.Tensor(generated)
    lpf_gen, f_gen, _ = mfn_decomposition_2D(generated, 0.5)
    lpf_true, f_true, _ = mfn_decomposition_2D(rp_around, 0.5)

    mse_matrix, cosine_matrix = f_mse(f_true, lpf_true, f_gen, lpf_gen)
    mse_matrix, cosine_matrix = mse_matrix.cpu().numpy(), cosine_matrix.cpu().numpy()
    lpf_gen, lpf_true = lpf_gen.cpu().numpy(), lpf_true.cpu().numpy()
    best_idx = int(np.argmin(mse_matrix))
    lpf_best = 0.5 * (lpf_gen[0] + lpf_true[best_idx])   # -> (L,)
    lpf_best = (lpf_best > 0).sum()  # sum of activated cells (superior to zero, where the signal stands)

    psnr_active = psnr_on_active_subset(
        x=generated[0], y=rp_around[best_idx],
        mx=lpf_gen[0],   my=lpf_true[best_idx],
        maxrp=maxrp, region="union", thr=0.0
    )
    mse_matrix = np.clip(mse_matrix, a_min=1e-12, a_max=30.)  # avoid div by zero in psnr
    rcs_diff = float(abs(generated.mean() - rp_around[best_idx].mean()).item())
    return psnr_active, np.max(cosine_matrix), np.min(mse_matrix), lpf_best, rcs_diff

def psnr_on_active_subset(x, y, mx, my, maxrp, region="union", thr=0.0, eps=1e-12):
    """
    x, y : (L,) HRRP (dé-normalisés, même échelle que maxrp)
    mx,my: (L,) LPF/masques (ex: lpf_gen, lpf_true)
    region: "union" | "intersection" | "true" | "pred"
    thr   : seuil pour activer (ex: 0.0 ou 1e-6)
    """
    ax = mx > thr
    ay = my > thr
    if region == "union":
        mask = ax | ay
    elif region == "intersection":
        mask = ax & ay
    elif region == "true":
        mask = ax
    elif region == "pred":
        mask = ay
    else:
        raise ValueError(region)
    x, y = x.squeeze(), y.squeeze()
    n = mask.sum()
    if n == 0:
        return float("nan")  # ou 100.0 selon ton choix
    mse_active = ((x - y)[mask]**2).mean()   # ← MSE standard, mais sur le sous-ensemble actif
    psnr = 20.0 * torch.log10(
        torch.tensor(maxrp, dtype=x.dtype, device=x.device) / torch.sqrt(mse_active + eps)
    )
    return float(psnr.item())

def load_scheduler_from_config(config):
    sched_params = config["scheduler"]
    if config["scheduler"]["name"] == "learnable":
        scheduler = LearnableScheduler(config["num_timesteps"], sched_params["hidden_dim"], sched_params["clipping_value"])
    elif config["scheduler"]["name"] == "cosine":
        scheduler = CosineScheduler(config["num_timesteps"], sched_params["s"], sched_params["clipping_value"])
    return scheduler

def get_save_path_create_folder(config, seed):
    save_path = config["figure_path"].split(" ")[0] + str(date.today()) + "/" + config["figure_path"].split(" ")[1]+"_seed"+str(seed)
    os.makedirs(save_path, exist_ok=True)
    dirs = [int(i) for i in os.listdir(save_path)]
    if len(dirs) == 0:
        z = 0
    else:
        z = np.max(dirs) + 1
    save_path = save_path + "/" + str(z)
    os.makedirs(save_path, exist_ok=True)

    return save_path

def unnormalize_hrrp(hrrp, min_rp, max_rp):
    hrrp = (hrrp + 1) / 2  # Back to [0, 1]
    hrrp = hrrp * (max_rp - min_rp) + min_rp
    return hrrp

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)

    return config

def get_means_stddevs(dataloader):
    means = []
    stddevs = []
    for images, _ in dataloader:
        for img in images:
            means.append(img.float().mean(dim=(0, 2, 3)))
            stddevs.append(img.float().std(dim=(0, 2, 3)))
    means, stddevs = np.array(means), np.array(stddevs)
    means = torch.Tensor(means).mean(dim=0)
    stddevs = torch.Tensor(stddevs).mean(dim=0)
    return means, stddevs


def test_dfva_in_rng(df_mmsi_va, rng1, rng2, i):
    if i != 1:
        test = (((df_mmsi_va > rng1[0]) & (df_mmsi_va < rng1[1])).any() or (
                (df_mmsi_va > rng2[0]) & (df_mmsi_va < rng2[1])).any())
    else:
        test = (((df_mmsi_va > rng1[0]) | (df_mmsi_va < rng1[1])).any() or (
                (df_mmsi_va > rng2[0]) & (df_mmsi_va < rng2[1])).any())
    return test

def mask_dfva_in_rng(dfva, rng1, rng2, i):
    if i != 1:
        mask = ((dfva > rng1[0]) & (dfva < rng1[1])) | ((dfva > rng2[0]) & (dfva < rng2[1]))
    else:
        mask = ((dfva > rng1[0]) | (dfva < rng1[1])) | ((dfva > rng2[0]) & (dfva < rng2[1]))
    return mask

def remove_zero_padding_color(image):
    # Trouver les indices des bords non nuls (en regardant les 3 canaux)
    rows_nonzero = np.where(np.any(image != 0, axis=(1, 2)))[0]
    cols_nonzero = np.where(np.any(image != 0, axis=(0, 2)))[0]
    # Délimiter l'image à ces indices
    cropped_image = image[rows_nonzero[0]:rows_nonzero[-1] + 1, cols_nonzero[0]:cols_nonzero[-1] + 1, :]
    return cropped_image

def resize_and_pad(image, target_shape):
    # Supprimer les zéros initiaux
    cropped_image = remove_zero_padding_color(image)

    # Calculer le facteur de mise à l'échelle pour conserver les proportions
    scale_factor = min(target_shape[0] / cropped_image.shape[0], target_shape[1] / cropped_image.shape[1])
    new_shape = (int(cropped_image.shape[0] * scale_factor), int(cropped_image.shape[1] * scale_factor), 3)
    resized_image = skimage.transform.resize(cropped_image, new_shape, anti_aliasing=True,
                                             preserve_range=True).astype(cropped_image.dtype)

    # Calculer le padding pour centrer l'image redimensionnée
    pad_height = (target_shape[0] - new_shape[0]) // 2
    pad_width = (target_shape[1] - new_shape[1]) // 2

    # Appliquer le padding avec les zéros autour de l'objet
    padded_image = np.pad(resized_image,
                          ((pad_height, target_shape[0] - new_shape[0] - pad_height),
                           (pad_width, target_shape[1] - new_shape[1] - pad_width),
                           (0, 0)),  # Ne pas ajouter de padding sur la dimension de couleur
                          mode='constant', constant_values=0)

    return padded_image


def detect_ship(binary_mask, aber=False):
    """
    Detects ship regions using erosion and dilation on a binary mask.
    Uses first rising edge (1) and last falling edge (-1) to define start and end.
    """
    if binary_mask.ndim == 1:
        binary_mask = binary_mask.reshape(1, -1)

    binary_mask = binary_mask.float()
    detected = apply_dilation_erosion(binary_mask)
    
    # Compute changes: rising and falling edges
    diff = torch.diff(detected.int(), dim=1)
    changes = torch.zeros_like(detected, dtype=torch.int32)
    changes[:, 1:] = diff

    starts = torch.full((detected.shape[0],), -1, dtype=torch.int32)
    ends = torch.full((detected.shape[0],), -1, dtype=torch.int32)

    for i in range(changes.shape[0]):
        change_i = changes[i]
        rising_edges = torch.where(change_i == 1)[0]
        falling_edges = torch.where(change_i == -1)[0]

        if len(rising_edges) > 0 and len(falling_edges) > 0:
            starts[i] = rising_edges[0].item()
            ends[i] = falling_edges[-1].item()
        else:
            # fallback: max value in smoothed
            max_idx = torch.argmax(binary_mask[i])
            starts[i] = max(0, max_idx - 1)
            ends[i] = min(binary_mask.shape[1] - 1, max_idx + 1)

    if aber:
        ships_pos = [list(range(starts[i], ends[i])) for i in range(len(starts))]
        return ships_pos
    else:
        lengths = ends - starts
        return lengths, starts, ends

def apply_dilation_erosion(x, kernel_size=15):
    # Dilation
    dilation = nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size // 2)
    dilated = dilation(x)

    # Erosion (via negative trick)
    eroded = -dilation(-dilated)
    return eroded

def uniform_filter_1d(signal, kernel_size=11):
    kernel = torch.ones(kernel_size)/kernel_size
    kernel = kernel.view(1, 1, -1).to(signal.device)
    smoothed_signal = F.conv1d(signal.unsqueeze(1), kernel, padding=int(kernel_size//2)).squeeze(1)
    return smoothed_signal

def get_df_RP_length(df, tresh=0.25, return_first_last=False, kernel_size=11):
    """
    Computes ship length using uniform filter-based detection.
    Uses first 1 and last -1 edge in the smoothed binary mask.
    """
    if isinstance(df, pd.DataFrame):
        global selectRP  # assuming it's set externally
        values = df[selectRP].values
        signal = torch.tensor(values, dtype=torch.float32)
    else:
        signal = df.float() if isinstance(df, torch.Tensor) else torch.tensor(df, dtype=torch.float32)

    smoothed = uniform_filter_1d(signal, kernel_size=kernel_size)

    # Compute threshold per signal
    if smoothed.ndim == 2:
        tresh_vals = tresh * torch.max(smoothed, dim=1, keepdim=True)[0]
        binary_mask = smoothed > tresh_vals
    else:
        tresh_val = tresh * torch.max(smoothed)
        binary_mask = (smoothed > tresh_val).unsqueeze(0)

    lengths, starts, ends = detect_ship(binary_mask)

    if not return_first_last:
        return lengths if smoothed.ndim != 1 else lengths[0]
    else:
        return (lengths, starts, ends) if smoothed.ndim != 1 else (lengths[0], starts[0], ends[0])
        

def get_expected_len(length, width, va):
    return abs(np.cos(va))*length + abs(np.sin(va))*width


def gaussian_filter_1d(signal, kernel_size=17, sigma=1.5):
    """
    Applies a 1D Gaussian filter using a differentiable convolution.
    """
    # Create Gaussian kernel
    kernel = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gaussian_kernel = torch.exp(- (kernel ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize

    # Convert to 3D kernel for 1D convolution (batch, channels, width)
    gaussian_kernel = gaussian_kernel.view(1, 1, -1).to(signal.device)
    
    # Apply convolution (keep input shape)
    smoothed_signal = F.conv1d(signal.unsqueeze(1), gaussian_kernel, padding=int(kernel_size // 2)).squeeze(1)
    return smoothed_signal

def mfn_decomposition_2D(RP, sigma, kernel_size=17):
    RP = RP.squeeze()
    if RP.ndim == 1:
        RP = RP.unsqueeze(0)
    num_signals, signal_length = RP.shape

    N = RP.shape[-1]
    mask = RP.new_ones(N)
    mask[185:] = 0.
    RP = RP * mask  # pas d'in-place, graphe propre

    # Get signal lengths and boundaries for all signals
    lrp, first, last = get_df_RP_length(RP, tresh=0.5, return_first_last=True)

    # Create indices as a 2D matrix: shape (num_signals, signal_length)
    indices = torch.tile(torch.arange(signal_length), (num_signals, 1))  # Shape (num_signals, signal_length)

    # --- Compute m component (Mean inside first:last) ---
    in_range = (indices >= first[:, None]) & (indices < last[:, None])  # Boolean mask
    means = torch.sum(RP * in_range, axis=1) / (torch.sum(in_range, axis=1)+1e-2)  # Compute mean only in range

    lpf = torch.zeros_like(RP)
    lpf[in_range] = torch.repeat_interleave(means[:, None], signal_length, axis=1)[in_range]  # Assign mean where in range

    # --- Compute mask ---
    mask = torch.ones_like(RP)

    # Left side mask
    left_mask = indices < first[:, None]  # Boolean mask for left side
    mask[left_mask] = torch.exp(2*(indices - first[:, None]) / (lrp[:, None] / 3))[left_mask]

    # Right side mask
    right_mask = indices >= last[:, None]  # Boolean mask for right side
    mask[right_mask] = torch.exp(2*(last[:, None] - indices) / (lrp[:, None] / 3))[right_mask]
    
    # --- Compute f component ---
    f_comp = gaussian_filter_1d(RP, kernel_size,  sigma) * mask

    # --- Compute n component ---
    n_comp = RP - f_comp

    return lpf, f_comp, n_comp

def f_mse(fx, mx, fy, my):
    if fy.shape[0] > fx.shape[0]:
        fx, mx = fx.repeat(fy.shape[0], 1), mx.repeat(fy.shape[0], 1)
    elif fy.shape[0] < fx.shape[0]:
        fy, my = fy.repeat(fx.shape[0], 1), my.repeat(fx.shape[0], 1)
    assert fx.shape[0] == fy.shape[0] and mx.shape[0] == fx.shape[0]
    num_signals = fx.shape[0]
    mse_matrix = torch.zeros((num_signals))
    cosine_matrix = torch.zeros((num_signals))
    for i in range(num_signals):
        if (mx[i]>0).sum() > (my[i]>0).sum(): # mi est plus large que mj, on veut donc élargir mj avec la même valeur 
            mi = mx[i]
            mj = my[i]
            mj[mi>0] = mj.max()                
        else:
            mj = my[i]
            mi = mx[i]
            mi[mj>0] = mi.max()
        if (mi>0).sum() == 0:
            fmi, lmi = 0, fx.shape[1]
        else:
            fmi, lmi = torch.argwhere(mi>0)[0][0],  torch.argwhere(mi>0)[-1][0]+1
        mse_matrix[i] = torch.mean((fx[i] - fy[i]) ** 2)/(((mx[i]>0).sum()+(my[i]>0).sum())/2)
        cosine_matrix[i] = cosine_similarity((fx[i]-mi)[fmi:lmi].reshape(1, -1), (fy[i]-mj)[fmi:lmi].reshape(1, -1)).item()
    return mse_matrix, cosine_matrix

"""     !!!     Outlier removing and plotting functions     !!!     """

def visualize_removance(function, HRRPs, df, *args):
    abertest1 = function(HRRPs, *args)
    print("Number of data removed by function {} : ".format(function.__name__), abertest1.sum())
    num_data = abertest1.sum()
    df_aber = df[abertest1]

    aberplot = HRRPs[abertest1]

    hrrps = np.zeros((10, 4, 200))
    little_title = np.zeros((10, 4))
    for i in range(10):
        for j in range(4):
            a = np.random.randint(0, num_data//(4*12))
            hrrps[i, j] = aberplot[num_data//12 * i + num_data//(4*12) * j + a]
            little_title[i, j] = df_aber.iloc[num_data//12 * i + num_data//(4*12) * j + a]["mmsi"]
    plot_HRRP(hrrps, function.__name__, little_title, "")
    return abertest1


def visualize_traj_removance(function, df, key, *args):
    abertest1 = function(df, *args)
    print("Number of data removed by function {} : ".format(function.__name__), abertest1.sum())
    df_aber = df[abertest1]
    tn_per_mmsi=[]
    for mmsi in df_aber[key].unique():
        df_aber_mmsi = df_aber[df_aber[key]==mmsi]
        if len(df_aber_mmsi)>40:
            tn_per_mmsi.append(mmsi)
        if len(tn_per_mmsi)>=60:
            break

    tn_per_mmsi = np.array(tn_per_mmsi)
    if tn_per_mmsi.shape[0]<60:
        tn_per_mmsi = tn_per_mmsi[:(tn_per_mmsi.shape[0]//6)*6]
    plot_trajs(df, tn_per_mmsi, 1, key, y_size=6)
    return abertest1


"""     !!!         Plot functions       !!!     """

def show_HRRPs(df, key1, value, key="viewing_angle", range=[0., 6.28], random=False, num_samples=40, rows=5, cols=8, save=False):
    if not random:
        df_to_sample = df[df[key1] == value]
        if range[0] < range[1]:
            df_to_sample = df_to_sample[(df_to_sample[key] >= range[0]) & (df_to_sample[key] <= range[1])]
        else:
            df_to_sample = df_to_sample[(df_to_sample[key] >= range[0]) | (df_to_sample[key] <= range[1])]
        num_samples = min(num_samples, len(df_to_sample))
    else:
        df_to_sample=df
        num_samples = min(num_samples, len(df_to_sample))
    # Sample 40 random rows from the DataFrame
    sampled_df = df_to_sample.sample(n=num_samples)
    df_to_plot = sampled_df[selectRP]
    print(df_to_plot.shape)

    # Calculate the mean of each HRRP and sort by this mean
    if random:
        mean = df_to_plot.apply(np.mean, axis=1)
        df_to_plot = pd.concat([df_to_plot, pd.DataFrame({"to_sort": mean})], axis=1)
    else:
        key_values = sampled_df[key]
        key_values_df = pd.DataFrame({"to_sort": key_values})
        df_to_plot = pd.concat([df_to_plot, key_values_df], axis=1)

    sorted_df = df_to_plot.sort_values(by='to_sort')
    sorted_df_idx = sorted_df.index
    if not random:
        little_title = np.round(sorted_df['to_sort'].to_numpy().reshape((rows, cols)), 2)
    else:
        little_title = sorted_df_idx.to_numpy().reshape((rows, cols))
    sorted_df = sorted_df.drop(columns='to_sort').to_numpy()
    sorted_df = np.reshape(sorted_df, (rows, cols, 200))

    # Plot the HRRPs in a grid
    plot_HRRP(sorted_df, 'Randomly sampled HRRPs', little_title, 'random_HRRPs.png', save=save)


def plot_one_traj(df, axs=1, j=1, oneo=7, oneoangle=14, time=True, angle=True):
    select1o5 = list(np.linspace(0, len(df) - 1, len(df) // oneo))

    X = df.X
    if oneo > 1:
        X = X.iloc[select1o5]
    X = X
    Y = df["Y"]
    if oneo > 1:
        Y = Y.iloc[select1o5]
    Y = Y

    u = np.diff(X)
    v = np.diff(Y)
    pos_x = X[:-1] + u / 2
    pos_y = Y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2) + 1e-4

    if type(axs) == np.ndarray:
        axs[0, j].plot(X, Y, marker="o")
        axs[0, j].plot(X.iloc[0], Y.iloc[0], marker="o", color="red")
        axs[0, j].plot(X.iloc[-1], Y.iloc[-1], marker="o", color="green")
        axs[0, j].quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        axs[0, j].scatter(0, 0, marker="x", linewidth=1.5, c="r")
        if "heading" in df.columns:
            for i in range(len(df.heading)):
                axs[0, j].text(X.iloc[oneoangle//oneo * i], Y.iloc[oneoangle//oneo * i], np.round(df.heading.iloc[oneoangle//oneo*i], 2))
        axs[0, j].text(0, 0, "radar")
        if time:
            axs[0, j].text(X.iloc[0], Y.iloc[0], str(df.iloc[0]["robin_timestamp"]))
            axs[0, j].text(X.iloc[-1], Y.iloc[-1], str(df.iloc[-1]["robin_timestamp"]))

    else:
        plt.plot(X, Y, marker="o")
        plt.plot(X.iloc[0], Y.iloc[0], marker="o", color="red")
        plt.plot(X.iloc[-1], Y.iloc[-1], marker="o", color="green")
        plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        plt.scatter(0, 0, marker="x", linewidth=1.5, c="r")
        if "heading" in df.columns and angle:
            for i in range(len(df.heading)//oneoangle):
                plt.text(X.iloc[oneoangle//oneo * i], Y.iloc[oneoangle//oneo * i], np.round(df.heading.iloc[oneoangle//oneo*i], 2))
        plt.text(0, 0, "radar")
        if time:
            plt.text(X.iloc[0], Y.iloc[0], str(df.iloc[0]["robin_timestamp"]))
            plt.text(X.iloc[-1], Y.iloc[-1], str(df.iloc[-1]["robin_timestamp"]))
        plt.show()
        return None
    return axs


def plot_dist_two_traj(df1, df2, axs=1, j=1, oneo=7, time=True):
    select1o5 = list(np.linspace(0, len(df1) - 1, len(df1) // oneo))

    X = df1.X
    if oneo > 1:
        X = X.iloc[select1o5]
    X = X
    Y = df1["Y"]
    if oneo > 1:
        Y = Y.iloc[select1o5]
    Y = Y

    u = np.diff(X)
    v = np.diff(Y)
    pos_x = X[:-1] + u / 2
    pos_y = Y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2) + 1e-4


    if type(axs) == np.ndarray:
        axs[0, j].plot(X, Y, marker="o")
        axs[0, j].plot(X.iloc[0], Y.iloc[0], marker="o", color="red")
        axs[0, j].plot(X.iloc[-1], Y.iloc[-1], marker="o", color="green")
        axs[0, j].quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        if time:
            axs[0, j].text(X.iloc[0], Y.iloc[0], str(df1.iloc[0]["robin_timestamp"]))
            axs[0, j].text(X.iloc[-1], Y.iloc[-1], str(df1.iloc[-1]["robin_timestamp"]))

    else:
        plt.plot(X, Y, marker="o")
        plt.plot(X.iloc[0], Y.iloc[0], marker="o", color="red")
        plt.plot(X.iloc[-1], Y.iloc[-1], marker="o", color="green")
        plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        if time:
            plt.text(X.iloc[0], Y.iloc[0], str(df1.iloc[0]["robin_timestamp"]))
            plt.text(X.iloc[-1], Y.iloc[-1], str(df1.iloc[-1]["robin_timestamp"]))

        select1o5 = list(np.linspace(0, len(df2) - 1, len(df2) // oneo))

        X = df2.X
        if oneo > 1:
            X = X.iloc[select1o5]
        X = X
        Y = df2["Y"]
        if oneo > 1:
            Y = Y.iloc[select1o5]
        Y = Y

        u = np.diff(X)
        v = np.diff(Y)
        pos_x = X[:-1] + u / 2
        pos_y = Y[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2) + 1e-4

        if type(axs) == np.ndarray:
            axs[0, j].plot(X, Y, c="b", marker="o")
            axs[0, j].quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
            if time:
                axs[0, j].text(X.iloc[0], Y.iloc[0], str(df2.iloc[0]["robin_timestamp"]))
                axs[0, j].text(X.iloc[-1], Y.iloc[-1], str(df2.iloc[-1]["robin_timestamp"]))

        else:
            plt.plot(X, Y, c="r", marker="o")
            plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
            if time:
                plt.text(X.iloc[0], Y.iloc[0], str(df2.iloc[0]["robin_timestamp"]))
                plt.text(X.iloc[-1], Y.iloc[-1], str(df2.iloc[-1]["robin_timestamp"]))
            plt.show()
        return None
    return axs


def plot_predicted_traj(df, axs=1, j=1, oneo=5):
    select1o5 = list(np.linspace(0, len(df) - 1, len(df) // oneo))

    X = df["X"]
    if oneo>1:
        X = X.iloc[select1o5]
    X = X
    Y = df["Y"]
    if oneo>1:
        Y = Y.iloc[select1o5]
    Y = Y

    u = np.diff(X)
    v = np.diff(Y)
    pos_x = X[:-1] + u / 2
    pos_y = Y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2) + 1e-4

    Xhat = df["xHat"]
    if oneo > 1:
        Xhat = Xhat.iloc[select1o5]
    Xhat = Xhat
    Yhat = df["yHat"]
    if oneo > 1:
        Yhat = Yhat.iloc[select1o5]
    Yhat = Yhat

    uhat = np.diff(Xhat)
    vhat = np.diff(Yhat)
    pos_xhat = X[:-1] + u / 2
    pos_yhat = Y[:-1] + v / 2
    normhat = np.sqrt(u ** 2 + v ** 2) + 1e-4


    if type(axs) == np.ndarray:
        axs[0, j].plot(X, Y, marker="o", c="g")
        axs[0, j].quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        axs[0, j].scatter(0, 0, marker="x", linewidth=1.5, c="r")
        axs[0, j].text(0, 0, "radar")
        axs[0, j].text(X.iloc[0], Y.iloc[0], str(df.iloc[0]["robin_timestamp"]))
        axs[0, j].text(X.iloc[-1], Y.iloc[-1], str(df.iloc[-1]["robin_timestamp"]))
        axs[0, j].plot(Xhat, Yhat, marker="o")
        axs[0, j].quiver(pos_xhat, pos_yhat, uhat / normhat, vhat / normhat, angles="xy", zorder=5, pivot="mid")

    else:
        plt.plot(X, Y, marker="o", c="g")
        plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
        plt.scatter(0, 0, marker="x", linewidth=1.5, c="r")
        plt.text(0, 0, "radar")
        plt.text(X.iloc[0], Y.iloc[0], str(df.iloc[0]["robin_timestamp"]))
        plt.text(X.iloc[-1], Y.iloc[-1], str(df.iloc[-1]["robin_timestamp"]))
        plt.plot(Xhat, Yhat, marker="o")
        plt.quiver(pos_xhat, pos_yhat, uhat / normhat, vhat / normhat, angles="xy", zorder=5, pivot="mid")
        plt.show()
        return None
    return axs


def plot_trajs(df, tnpermmsi, little_title, key, one_over=5, y_size=6):
    fig, axs = plt.subplots(len(tnpermmsi)//6, y_size, figsize=(24, 3*(len(tnpermmsi))//6), layout='constrained', squeeze=False)
    i = 0
    tnpermmsi = np.array(tnpermmsi)
    tnpermmsi = tnpermmsi.reshape((-1, 6))
    for tnlist in tnpermmsi:
        j = 0
        for tn in tnlist:
            shipTN = df[df[key] == tn]
            select1o = list(np.linspace(0, len(shipTN)-1, len(shipTN) // one_over))

            X = shipTN["X"]
            X = X.iloc[select1o]
            Y = shipTN["Y"]
            Y = Y.iloc[select1o]

            u = np.diff(X)
            v = np.diff(Y)
            pos_x = X[:-1] + u / 2
            pos_y = Y[:-1] + v / 2
            norm = np.sqrt(u ** 2 + v ** 2) + 1e-4

            axs[i, j].plot(X, Y, marker="o")
            axs[i, j].plot(X.iloc[0], Y.iloc[0], marker="o", color="red")
            axs[i, j].plot(X.iloc[-1], Y.iloc[-1], marker="o", color="green")
            axs[i, j].quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
            axs[i, j].scatter(0, 0, marker="x", linewidth=1.5, c="r")
            axs[i, j].text(0, 0, "radar")
            if isinstance(little_title, np.ndarray):
                little_title = little_title.reshape((-1, 6))
                axs[i, j].set_title(
                    str(little_title[i, j]))
            if isinstance(little_title, list):
                little_title = np.array(little_title)
                little_title = little_title.reshape((-1, 6))
                axs[i, j].set_title(
                    str(little_title[i][j]))
            j += 1
        i += 1
    plt.show()


def get_var_plot(shipTN, select1o):
    X = shipTN["X"]
    X = X.iloc[select1o]
    Y = shipTN["Y"]
    Y = Y.iloc[select1o]

    u = np.diff(X)
    v = np.diff(Y)
    pos_x = X[:-1] + u / 2
    pos_y = Y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2) + 1e-4
    return X, Y, u, v, pos_x, pos_y, norm


def plot_comparison_match(rp, ais, tm, nt, little_title, little_titleais, one_over=5, y_size=4, save=False):
    fig, axs = plt.subplots(len(tm) // 2, y_size, figsize=(20, 3 * len(tm)), squeeze=False)
    i = 0
    tm, nt, little_title, little_titleais = np.array(tm), np.array(nt), np.array(little_title), np.array(
        little_titleais)
    assert tm.shape == nt.shape, "nt and tm must have same shape, tm shape {}, tn shape {}".format(tm.shape, nt.shape)
    tm, nt, little_title, little_titleais = tm.reshape((-1, y_size // 2)), nt.reshape(
        (-1, y_size // 2)), little_title.reshape((-1, y_size // 2)), little_titleais.reshape((-1, y_size // 2))
    for ntlist in nt:
        j = 0
        z = 0
        for nt_ship in ntlist:
            tmship = ais[ais["track_mmsi"] == tm[i, z]]
            shipTN = rp[rp["new_tracks"] == nt_ship]
            select1o = list(np.linspace(0, len(shipTN) - 1, len(shipTN) // one_over))
            select1oais = list(np.linspace(0, len(tmship) - 1, len(tmship) // one_over))
            X, Y, u, v, pos_x, pos_y, norm = get_var_plot(shipTN, select1o)
            Xais, Yais, uais, vais, pos_xais, pos_yais, normais = get_var_plot(tmship, select1oais)
            xmax, xmin, ymax, ymin = max(X.max(), Xais.max()), min(X.min(), Xais.min()), max(Y.max(), Yais.max()), min(
                Y.min(), Yais.min())
            axs[i, j + 1].text(tmship.X.iloc[0], tmship.Y.iloc[0], tmship.robin_timestamp.iloc[0], fontsize=8,
                               ha='left', va='bottom')
            axs[i, j + 1].text(tmship.X.iloc[-1], tmship.Y.iloc[-1], tmship.robin_timestamp.iloc[-1], fontsize=8,
                               ha='left', va='bottom')
            axs[i, j + 1].plot(Xais, Yais, marker="o")
            axs[i, j + 1].plot(Xais.iloc[0], Yais.iloc[0], marker="o", color="red")
            axs[i, j + 1].plot(Xais.iloc[-1], Yais.iloc[-1], marker="o", color="green")
            axs[i, j + 1].quiver(pos_xais, pos_yais, uais / normais, vais / normais, angles="xy", zorder=5, pivot="mid")
            axs[i, j + 1].scatter(0, 0, marker="x", linewidth=1.5, c="r")
            axs[i, j + 1].text(0, 0, "radar")
            if isinstance(little_title, np.ndarray):
                axs[i, j + 1].set_title(
                    str(little_titleais[i, z]))
            if isinstance(little_title, list):
                axs[i, j + 1].set_title(
                    str(little_titleais[i][z]))
            axs[i, j + 1].set_xlim(min(xmin, 0), max(xmax, 0))
            axs[i, j + 1].set_ylim(min(ymin, 0), max(ymax, 0))

            axs[i, j].text(shipTN.X.iloc[0], shipTN.Y.iloc[0], shipTN.robin_timestamp.iloc[0], fontsize=8, ha='left',
                           va='bottom')
            axs[i, j].text(shipTN.X.iloc[-1], shipTN.Y.iloc[-1], shipTN.robin_timestamp.iloc[-1], fontsize=8, ha='left',
                           va='bottom')
            axs[i, j].plot(X, Y, marker="o")
            axs[i, j].plot(X.iloc[0], Y.iloc[0], marker="o", color="red")
            axs[i, j].plot(X.iloc[-1], Y.iloc[-1], marker="o", color="green")
            axs[i, j].quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
            axs[i, j].scatter(0, 0, marker="x", linewidth=1.5, c="r")
            axs[i, j].text(0, 0, "radar")
            if isinstance(little_title, np.ndarray):
                axs[i, j].set_title(
                    str(little_title[i, z]))
            if isinstance(little_title, list):
                axs[i, j].set_title(
                    str(little_title[i][z]))
            axs[i, j].set_xlim(min(xmin, 0), max(xmax, 0))
            axs[i, j].set_ylim(min(ymin, 0), max(ymax, 0))

            j += 2
            z += 1
        i += 1
    plt.tight_layout()
    if save:
        plt.savefig("./figures/comparison_match")
    plt.show()

def plot_HRRP(hrrp, global_title, little_title, file, save=False):
    assert len(hrrp.shape) == 3, "shape of hrrps must be 3, size you tried {}".format(hrrp.shape)
    fig, axs = plt.subplots(hrrp.shape[0], hrrp.shape[1], figsize=(16, 3*hrrp.shape[0]), layout='constrained')
    for i in range(hrrp.shape[0]*hrrp.shape[1]):
        axs[i//hrrp.shape[1], i%hrrp.shape[1]].plot(range(1, 201), hrrp[i//hrrp.shape[1], i%hrrp.shape[1]])
        if isinstance(little_title, np.ndarray):
            axs[i//hrrp.shape[1], i%hrrp.shape[1]].set_title(str(little_title[i//hrrp.shape[1], i%hrrp.shape[1]]))
        if isinstance(little_title, list):
            axs[i // hrrp.shape[1], i % hrrp.shape[1]].set_title(
                str(little_title[i // hrrp.shape[1]][i % hrrp.shape[1]]))
        fig.suptitle(global_title)
    if save:
        plt.savefig(file)
    else:
        plt.show()


def plot_traj_with_time(df, mode="RP", oneo=5):
    select1o = list(np.linspace(0, len(df) - 1, len(df) // oneo))
    df = df.iloc[select1o]
    colors = mcolors.CSS4_COLORS
    keys = list(mcolors.CSS4_COLORS)
    keys = np.array(keys)
    nb_20min = (df.unix_seconds.max() - df.unix_seconds.min())//1200
    max_t = df.unix_seconds.max()
    factor = keys.shape[0]//nb_20min
    key = keys[np.array((factor*(max_t - df.unix_seconds)//1200)).astype(int)]
    plt.scatter(df.X, df.Y, s=4, marker="o", linewidth=0.5, c=[colors[i] for i in key])
    plt.text(df.X.iloc[0], df.Y.iloc[0], df.robin_timestamp.iloc[0])
    plt.text(df.X.iloc[-1], df.Y.iloc[-1], df.robin_timestamp.iloc[-1])
    if mode=="RP":
        plt.title("Track number : {}".format(df.new_tracks.iloc[0]))
    elif mode=="AIS":
        plt.title("MMSI track : {}".format(df.track_mmsi.iloc[0]))
    plt.show()


### =====  Translation functions ===== ###

# --- Gauss 1D (semblable à uniform_filter_1d mais avec noyau gaussien) ---
def gaussian_filter_1d(signal: torch.Tensor, kernel_size=21, sigma=3.0):
    """
    signal: (B, L) ou (L,)  -> retourne (B, L)
    """
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)  # (1, L)
    # noyau gaussien
    x = torch.arange(kernel_size, device=signal.device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    # conv
    smoothed = F.conv1d(signal.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)
    return smoothed  # (B, L)

# --- Détection robuste début/fin via morphologie (tes fonctions adaptées torch) ---
def apply_dilation_erosion(x, kernel_size=15):
    dilation = torch.nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size // 2)
    dilated = dilation(x)
    eroded = -dilation(-dilated)
    return eroded

def detect_ship(binary_mask):
    """
    binary_mask: (B, L) bool/0-1 float
    Retourne: lengths, starts, ends (tensors int32)
    """
    if binary_mask.ndim == 1:
        binary_mask = binary_mask.reshape(1, -1)
    binary_mask = binary_mask.float()
    detected = apply_dilation_erosion(binary_mask)

    diff = torch.diff(detected.int(), dim=1)
    changes = torch.zeros_like(detected, dtype=torch.int32)
    changes[:, 1:] = diff

    B, L = detected.shape
    starts = torch.full((B,), -1, dtype=torch.int32)
    ends   = torch.full((B,), -1, dtype=torch.int32)

    for i in range(B):
        change_i = changes[i]
        rising_edges  = torch.where(change_i == 1)[0]
        falling_edges = torch.where(change_i == -1)[0]

        if len(rising_edges) > 0 and len(falling_edges) > 0:
            starts[i] = rising_edges[0].item()
            ends[i]   = falling_edges[-1].item()
        else:
            # fallback: autour du maximum
            max_idx = torch.argmax(binary_mask[i]).item()
            starts[i] = max(0, max_idx - 1)
            ends[i]   = min(L - 1, max_idx + 1)

    lengths = ends - starts
    return lengths, starts, ends

# --- Traduction (alignement) basée sur le début détecté sur signal gaussien ---
def translate_zero_gaussian(df: pd.DataFrame,
                            align_to: int = 15,
                            thresh: float = 0.33,
                            gauss_kernel_size: int = 21,
                            gauss_sigma: float = 3.0,
                            morph_kernel_size: int = 15):
    """
    Remplace translate_zero en utilisant un lissage gaussien + détection de début.
    - On lisse chaque HRRP
    - On construit un masque binaire > thresh * max(smooth)
    - On détecte le premier front montant (start)
    - On décale les lignes où start >= align_to pour aligner start sur 'align_to'
    """
    global selectRP  # mêmes colonnes que ta fonction d'origine
    hrrp = df[selectRP].to_numpy()              # (B, L) numpy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sig = torch.tensor(hrrp, dtype=torch.float32, device=device)  # (B, L)

    # 1) Lissage gaussien
    smooth = gaussian_filter_1d(sig, kernel_size=gauss_kernel_size, sigma=gauss_sigma)  # (B, L)

    # 2) Seuil relatif par ligne
    max_per_row = torch.max(smooth, dim=1, keepdim=True)[0].clamp(min=1e-12)
    binary_mask = (smooth > (thresh * max_per_row)).to(torch.float32)  # (B, L)

    # 3) Morphologie + premier/dernier edge
    _, starts, _ = detect_ship(binary_mask)

    # 4) Calcul du décalage: align_to - start
    starts = starts.to(torch.int64)  # (B,)
    shift_amount = align_to - starts  # positif: décalage à droite, négatif: à gauche
    mask = (starts >= align_to)       # même logique que ton masque original

    B, L = sig.shape
    shifted = torch.full_like(sig, torch.nan)

    # 5) Décalage par broadcasting
    row_idx = torch.arange(B, device=device)[mask]                    # indices de lignes à décaler
    col_idx = torch.arange(L, device=device).view(1, -1)              # (1, L)
    shifts  = shift_amount[mask].view(-1, 1).to(device)               # (b_mask, 1)
    tgt_idx = col_idx + shifts                                        # (b_mask, L)

    # validité des colonnes cibles
    valid = (tgt_idx >= 0) & (tgt_idx < L)
    # clamp pour indexer (on utilisera valid pour filtrer)
    tgt_idx_clamped = tgt_idx.clamp(0, L - 1)

    # gather/scatter manuel
    gathered_src = sig[row_idx.unsqueeze(1), col_idx]                 # (b_mask, L)
    # on initialise en NaN, puis on remplit seulement les positions valides
    shifted_rows = torch.full_like(gathered_src, torch.nan)
    shifted_rows[valid] = gathered_src[valid]

    # Scatter dans 'shifted'
    shifted[row_idx.unsqueeze(1), tgt_idx_clamped] = shifted_rows

    # 6) Remplir les NaN avec la valeur d'origine (comme ta version)
    nan_mask = torch.isnan(shifted)
    shifted[nan_mask] = sig[nan_mask]

    # 7) Remettre uniquement les lignes masquées, sinon on garde l’original
    out = sig.clone()
    out[mask] = shifted[mask]

    # 8) Écriture dans le DataFrame
    df.loc[mask.detach().cpu().numpy(), selectRP] = out[mask].detach().cpu().numpy()
    return df
