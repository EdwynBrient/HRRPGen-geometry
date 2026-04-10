import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from transformers import AutoModelForImageClassification
import pytorch_lightning as pl
from math import pi
from tqdm import tqdm
import numpy as np
import torchvision
import copy
import os
try:
    from .ema import EMA
    from .utils import *
    from .schedulers import CosineScheduler, QuadraticScheduler
except ImportError:  # pragma: no cover
    from ema import EMA
    from utils import *
    from schedulers import CosineScheduler, QuadraticScheduler
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import time


#
# class EMA:
#     def __init__(self, model: torch.nn.Module, decay: float = 0.9):
#         self.model = model
#         self.decay = decay
#         self.shadow = {
#             name: param.clone().detach()
#             for name, param in model.named_parameters() if param.requires_grad
#         }
#
#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name].data = (
#                     self.decay * self.shadow[name].data.to(param.device)
#                     + (1.0 - self.decay) * param.data
#                 )
#
#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 param.data = self.shadow[name].data
#
#     def store(self):
#         self.backup = {
#             name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad
#         }
#
#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 param.data = self.backup[name].data

class ResMasks(nn.Module):
    def __init__(self, inplanes, planes, scal_dim=None, tdim=64):
        super(ResMasks, self).__init__()
        self.scal_dim=scal_dim
        self.planes=planes
        self.block_1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(planes),
            nn.SiLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(planes),
            nn.SiLU(),
        )
        if scal_dim is not None:
            self.scal_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(tdim, planes),
            )

    def forward(self, x, scal_emb=None):
        residual = x
        out = self.block_1(x)

        if self.scal_dim is not None:
            out += self.scal_proj(scal_emb).reshape(-1, self.planes, 1, 1)

        out = self.block_2(x)
        out += residual
        return out
    
class DownsampleMask(nn.Module):
    """
    Downsample the mask's features
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size = 5, stride = 4, padding = 2)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.relu(self.layer(x))
    
class FullMask(nn.Module):
    def __init__(self, config, tdim):
        super().__init__()
        self.config = config
        self.scal_emb=config["conditionned"]["scal_emb"]
        self.num_res = config["conditionned"]["num_res"]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.resblocks = nn.ModuleList([])
        self.downblocks = nn.ModuleList([])
        num_hid=16
        for _ in range(self.num_res):
            self.resblocks.append(ResMasks(num_hid, num_hid, self.scal_emb, tdim))
            self.downblocks.append(DownsampleMask(num_hid, num_hid))
        self.convout = nn.Conv2d(num_hid, num_hid//2, kernel_size=5, stride=2, padding=2)
    
    def forward(self, x:torch.Tensor, scal:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for i in range(self.num_res):
            x = self.resblocks[i](x, scal) if self.scal_emb else self.resblocks[i](x)
            x = self.downblocks[i](x)
        return self.convout(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, input_dim, condition_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(condition_dim, input_dim)
        self.value_proj = nn.Linear(condition_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim),
        )
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x, condition):
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_len]
            condition: Conditioning tensor of shape [batch_size, condition_len, condition_dim]

        Returns:
            Updated tensor of shape [batch_size, channels, seq_len]
        """
        # Step 1: Permute input tensor to [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)

        # Step 2: Project queries, keys, and values
        queries = self.query_proj(x)  # [batch_size, seq_len, input_dim]
        keys = self.key_proj(condition)  # [batch_size, condition_len, input_dim]
        values = self.value_proj(condition)  # [batch_size, condition_len, input_dim]

        # Step 3: Apply cross-attention
        attn_output, _ = self.attention(queries, keys, values)  # [batch_size, seq_len, input_dim]
        # Step 4: Add residual connection and normalize
        x = self.norm1(x + attn_output)

        # Step 5: Apply feedforward and residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)

        # Step 6: Permute back to [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        return x

def timestep_embedding(timesteps:torch.Tensor, dim:int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class EmbedBlock(nn.Module):
    """
    abstract class
    """
    def forward(self, x, temb, cemb):
        """
        abstract method
        """

class EmbedBlockUncond(nn.Module):
    """
    abstract class
    """
    def forward(self, x, temb):
        """
        abstract method
        """

class EmbedSequential(nn.Sequential, EmbedBlock, EmbedBlockUncond):
    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            elif isinstance(layer, EmbedBlockUncond):
                x = layer(x, temb)
            else:
                x = layer(x)
        return x
    
class ResBlock(EmbedBlock):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim:int, cdim:int, dropout:float, dbcond=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.dropout = dropout
        self.dbcond = dbcond

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        
        self.c1emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, out_ch),
        )
        
        if self.dbcond is not None:
            self.c2emb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cdim, out_ch),
            )

        self.block_2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
        )

        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()

    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None]
        latent += self.c1emb_proj(cemb[0])[:, :, None]
        if self.dbcond is not None:
            latent += self.c2emb_proj(cemb[1])[:, :, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
    
class ResBlockUncond(EmbedBlockUncond):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim:int, dropout:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.dropout = dropout

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            
        )
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()
    def forward(self, x:torch.Tensor, temb:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
    
class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

class EmbedLinear(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, ):
        super().__init__()
        self.layer1 = nn.Linear(in_ch, out_ch)
        self.gain = Gain()
        self.layernorm = nn.LayerNorm(out_ch)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.gain(self.layer1(x)))

class Upsample(nn.Module):
    """
    an upsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv1d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        output = self.layer(x)
        return output

class Downsample1d(nn.Module):
    """
    a downsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv1d(self.in_ch, self.out_ch, kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        return self.layer(x)

class ResUnet(nn.Module):
    def __init__(self, config):
        super(ResUnet, self).__init__()

        # Sinusoidal embedding
        self.n_steps = config["num_timesteps"]
        self.dtype = torch.float32
        self.config = config
        self.mod_ch = config["mod_ch"]
        self.ch_mul = config["ch_mul"]
        self.num_res_blocks = config["num_res"]
        self.conditionned= config["conditionned"]
        self.cond = self.conditionned["bool"]
        if self.cond:
            self.scal_emb = self.conditionned["scal_emb"]
            self.type = self.conditionned["type"]
            self.use_dims = self.conditionned.get("dims", True)
            self.use_va = self.conditionned.get("va", True)
        self.class_free = config["class_free"]["bool"]
        self.dropout = config["dropout"]
        if self.class_free:
            self.guidance = config["class_free"]["guidance"]
                           
        tdim = self.mod_ch * 4
        self.temb_layer = EmbedLinear(self.mod_ch, tdim)

        if self.cond:    
            if self.type in ("scalars", "both", "dims"):
                # by default we keep VA + dimensions; can be toggled via config
                scalar_inputs = (1 if self.use_va else 0) + (2 if self.use_dims else 0)
                if scalar_inputs == 0:
                    raise ValueError("At least one scalar (va or dimensions) must be enabled when conditioning on scalars.")
                self.scal_emb_layer = EmbedLinear(scalar_inputs*self.mod_ch, tdim)
            if self.type == "masks" or self.type=="both":
                self.make_masks_emb = FullMask(config, tdim)
                self.mask_emb_layer = EmbedLinear(784, tdim)
            cdim = tdim
        
        dbcond = None
        if self.type=="both":
            dbcond = 1    

        now_ch = self.ch_mul[0] * self.mod_ch
        chs = [now_ch]

        self.downblocks = nn.ModuleList([
                    EmbedSequential(nn.Conv1d(1, self.mod_ch, 3, padding=1))
                ])            


        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            for _ in range(self.num_res_blocks):
                if self.cond:
                    layers = [
                        ResBlock(now_ch, nxt_ch, tdim, cdim, self.dropout, dbcond),
                    ]
                else:
                    layers = [
                        ResBlockUncond(now_ch, nxt_ch, tdim, self.dropout),
                    ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(Downsample1d(now_ch, now_ch)))
                chs.append(now_ch)
        if self.cond:
            self.middleblocks = EmbedSequential(
                ResBlock(now_ch, now_ch, tdim, cdim, self.dropout, dbcond),
                ResBlock(now_ch, now_ch, tdim, cdim, self.dropout, dbcond)
            )
        else:
            self.middleblocks = EmbedSequential(
                ResBlockUncond(now_ch, now_ch, tdim, self.dropout),
                ResBlockUncond(now_ch, now_ch, tdim, self.dropout)
            )
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(self.num_res_blocks + 1):
                if self.cond:
                    layers = [
                        ResBlock(now_ch+chs.pop(), nxt_ch, tdim, cdim, self.dropout, dbcond),
                    ]
                else:
                    layers = [
                        ResBlockUncond(now_ch+chs.pop(), nxt_ch, tdim, self.dropout),
                    ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            nn.SiLU(),
            nn.Conv1d(now_ch, 1, 3, stride = 1, padding = 1)
        )

    def forward(self, x, t, c):  # x is (bs, in_c, size) t is (bs)
        temb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        n = x.shape[0]
        if self.cond:
            if self.type in ("scalars", "dims"):
                vars = c
            else:
                masks, vars = c
            vars = torch.concat([timestep_embedding(vars[:, i], self.mod_ch, 10) for i in range(vars.shape[1])], dim=1)
            if self.type!="masks":
                vars = self.scal_emb_layer(vars).reshape(n, -1)
            if self.type != "scalars" and self.type != "dims":
                if self.scal_emb:
                    mask1, mask2 = self.make_masks_emb(masks[0], vars).reshape(n, -1), self.make_masks_emb(masks[1], vars).reshape(n, -1)
                else:
                    mask1, mask2 = self.make_masks_emb(masks[0]).reshape(n, -1), self.make_masks_emb(masks[1]).reshape(n, -1)

                masks = torch.concat([mask1, mask2], dim=1)
                masks = self.mask_emb_layer(masks.reshape(n, -1))

            if self.type in ("scalars", "dims"):
                conds = [vars]  
            elif self.type =="masks":
                conds = [masks]
            else:
                conds = [masks, vars]

            if self.training:
                conds = [conds[i] + torch.randn_like(conds[i]) * self.conditionned["noise"] for i in range(len(conds))]
            if self.class_free:
                sample_guidance = np.random.uniform(size=n)
                if self.type == "both":
                    conds[0][sample_guidance > self.guidance] = 0
                    conds[1][sample_guidance > self.guidance] = 0
                else:
                    conds[0][sample_guidance > self.guidance] = 0

        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            if self.cond:
                h = block(h, temb, conds)
            else:
                h = block(h, temb, 0)
            hs.append(h)
        if self.cond:
            h = self.middleblocks(h, temb, conds)
        else:
            h = self.middleblocks(h, temb, 0)
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim = 1)
            if self.cond:
                h = block(h, temb, conds)
            else:
                h = block(h, temb, 0)
        h = h.type(self.dtype)
        return self.out(h)


class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, var_scheduler=CosineScheduler) -> None:
        super(DDPM, self).__init__()
        self.var_scheduler = var_scheduler
        self.num_timesteps = num_timesteps
        self.alphas_cumprod = self.var_scheduler.get_alpha_hat()
        self.alphas = self.var_scheduler.get_alphas()
        self.betas = self.var_scheduler.get_betas()
        self.network = network
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5  # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        """
        Add noise to the input data using dynamically computed alpha_hat.
        Args:
            x_start (torch.Tensor): Original data (e.g., images).
            x_noise (torch.Tensor): Noise to add.
            timesteps (torch.Tensor): Current timesteps for the batch.
        Returns:
            torch.Tensor: Noisy data.
        """
        # Get betas dynamically from the learned scheduler
        betas = self.var_scheduler().to(x_start.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(x_start.device)

        # Gather the necessary alpha_hat values for the given timesteps
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps])

        # Reshape for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)

        # Add noise
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * x_noise

    def reverse(self, x, t, c):
        # The network return the estimation of the noise we added
        return self.network(x, t, c)

    def step(self, model_output, timestep, sample, noise_override=None, use_prev_t_for_noise=False):
        """
        Perform one reverse diffusion step using dynamically computed values.
        Args:
            model_output (torch.Tensor): Predicted noise by the model.
            timestep (int): Current timestep.
            sample (torch.Tensor): Current noisy sample.
            noise_override (torch.Tensor, optional): Reuse a specific noise tensor instead of sampling a new one.
            use_prev_t_for_noise (bool): If True, use t-1 to scale the noise (useful to align with forward noising at t-1).
        Returns:
            torch.Tensor: Predicted sample at the previous timestep.
        """
        # Get dynamic betas and alphas
        betas = self.var_scheduler().to(sample.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(sample.device)

        # Ensure timestep is a tensor on the right device
        timestep_tensor = timestep.to(sample.device) if torch.is_tensor(timestep) else torch.tensor(timestep, device=sample.device)
        noise_timestep = timestep_tensor - 1 if use_prev_t_for_noise else timestep_tensor
        noise_timestep = torch.clamp(noise_timestep, min=0)

        # Gather values for the given timestep
        alpha_t = alphas[timestep_tensor]
        alpha_hat_t = alphas_cumprod[timestep_tensor]

        # Compute coefficients
        coef_eps_t = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
        coef_first = 1 / torch.sqrt(alpha_t)
        coef_eps_t = coef_eps_t.view(-1, 1, 1).to(sample.device)
        coef_first = coef_first.view(-1, 1, 1).to(sample.device)

        # Predict previous sample
        pred_prev_sample = coef_first * (sample - coef_eps_t * model_output)
        
        # Add noise for variance
        variance = 0
        has_noise = bool(torch.any(timestep_tensor > 0)) if torch.is_tensor(timestep_tensor) else timestep_tensor > 0
        if has_noise:
            noise = noise_override if noise_override is not None else torch.randn_like(model_output).to(sample.device)
            variance = torch.sqrt(betas[timestep_tensor]) * noise
        return pred_prev_sample + variance

class DDPMlight1D2D(pl.LightningModule):
    def __init__(self, ddpm, config, dataset, validation_indices, save_path, minmax):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.df = dataset.df
        self.df_va = dataset.old_va
        self.val_idx = validation_indices
        self.bs=30
        self.cond = self.config["conditionned"]["bool"]
        self.type = self.config["conditionned"]["type"]
        if self.type in ("scalars", "dims"):
            self.nb_id = 2
        else:
            self.nb_id = 3
        self.min_rp, self.max_rp = minmax
        self.ddpm = ddpm
        self.num_timesteps = config["num_timesteps"]
        self.inf_every = config["inf_every_n_epoch"]
        scheduler = load_scheduler_from_config(config)
        self.save_path = save_path
        self.loss = config["loss"]
        self.scheduler = scheduler
        self._last_infer_bucket = {"train": -1, "val": -1}  # pour inf_mode="step"
        self._last_infer_epoch  = {"train": -1, "val": -1}  # pour inf_mode="epoch"

    def _extract_scalars(self, vars: torch.Tensor) -> torch.Tensor:
        cond_cfg = self.config["conditionned"]
        use_va = cond_cfg.get("va", True)
        use_dims = cond_cfg.get("dims", True)
        scalars = []
        if use_va:
            scalars.append(vars[:, :, -3])
        if use_dims:
            scalars.extend([vars[:, :, -2], vars[:, :, -1]])
        if len(scalars) == 0:
            raise ValueError("No scalar features selected: enable at least one of conditionned.va or conditionned.dims.")
        return torch.concat(scalars, dim=1).float()

    def test_inf(self, mode: str) -> bool:
        if self.current_epoch == 0:
            return False
        assert mode in ("train", "val")
        if self.config["inf_mode"] == "step":
            k = int(self.config["inf_every_n_step"])
            if k <= 0:
                return False
            # global_step est incrémenté après l'optim step ; on anticipe le step courant
            next_step = self.global_step + 1
            bucket = next_step // k
            if next_step > 0 and bucket > self._last_infer_bucket[mode]:
                self._last_infer_bucket[mode] = bucket
                return True
            return False
        else:  # "epoch"
            k = int(self.inf_every)
            if k <= 0:
                return False
            # Déclenche une seule fois aux époques multiples de k (1-indexées)
            human_epoch = self.current_epoch + 1
            if human_epoch % k == 0 and self._last_infer_epoch[mode] != self.current_epoch:
                self._last_infer_epoch[mode] = self.current_epoch
                return True
            return False

    def roll_x0_from_xt(self, xt, t, c):
        """
        Recover x0 from xt in the DDPM framework.

        Args:
            xt: Noisy sample at timestep t, shape (batch_size, *data_dims).
            t: Current timestep as a tensor, shape (batch_size,).
            c: Conditioning information, if required by the network.

        Returns:
            x0: Approximation of the original data, same shape as xt.
        """
        with torch.no_grad():
            # Ensure `t` is a 1D tensor and move it to CPU for indexing
            t = t.view(-1).cpu()

            # Access precomputed sqrt terms
            sqrt_alpha_cumprod_t = self.ddpm.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(xt.device)  # (bs, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = self.ddpm.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(
                xt.device)  # (bs, 1, 1, 1)

            # Predict noise using the reverse process
            epsilon_theta = self.ddpm.reverse(xt, t.to(xt.device), c)

            # Recover x0 using the formula
            x0 = (xt - sqrt_one_minus_alpha_cumprod_t * epsilon_theta) / sqrt_alpha_cumprod_t
            return x0
    
    def compute_loss(self, mode, idx, generated):
        t_loss = [top_loss(self.df, idx[i].cpu().item(), generated[i], self.min_rp, self.max_rp, 0.04) for i in range(generated.shape[0])]
        tpsnr = [t[0] for t in t_loss]
        tcos_f = [t[1] for t in t_loss]
        tmse_f = [t[2] for t in t_loss]
        lpf = [t[3] for t in t_loss]
        rcs_f = [t[4] for t in t_loss]
        if mode != "test":
            self.log(f"top_psnr_{mode}", torch.mean(torch.tensor(tpsnr)), prog_bar=True)
            self.log(f"top_cosf_{mode}", torch.mean(torch.tensor(tcos_f)), prog_bar=True)
            self.log(f"top_msef_{mode}", torch.mean(torch.tensor(tmse_f)), prog_bar=True)
            self.log(f"top_rcs_{mode}", torch.mean(torch.tensor(rcs_f)), prog_bar=True)
        return torch.tensor(tpsnr), torch.tensor(tcos_f), torch.tensor(tmse_f), torch.tensor(lpf), torch.tensor(rcs_f)

    def forward(self, sample_batch):
        with torch.no_grad():
            self.ddpm.to(self.device)
            frames = []
            timesteps = list(range(1, self.ddpm.num_timesteps))[::-1]
            if self.type in ("scalars", "dims"):
                vars = sample_batch[0]
            else:
                masks, vars = sample_batch 
                masks = [masks[:self.bs, 0].to(self.device), masks[:self.bs, 1].to(self.device)]
            vars = vars[:self.bs]
            RP = vars[:, 0, :200].float()

            sample = torch.randn(vars.shape[0], 1, 200).float().to(self.device)
            scal = self._extract_scalars(vars)
            
            for i, t in enumerate(timesteps):
                time_tensor = (torch.ones(RP.shape[0], 1) * t).long().to(self.device).squeeze()
                if self.type in ("scalars", "dims"):
                    residual = self.ddpm.reverse(sample, time_tensor, scal)
                else:
                    residual = self.ddpm.reverse(sample, time_tensor, [masks, scal])
                sample = self.ddpm.step(residual, time_tensor[0], sample)

            for i in range(RP.shape[0]):
                frames.append(sample[i].detach().cpu().numpy())
        return frames, RP

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.ddpm.network.parameters(), lr=self.config["lr"])
        if self.config["cosine_annealing"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config["epochs"], eta_min=1e-8)
            return {
                'optimizer': opt,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # or 'step' if you want to update every batch
                    'frequency': 1        # how often to step (1 = every epoch)
                }
            }

        else:
            return {"optimizer": opt}
    
    def lambda_t(self, t):
        alpha_cumprod = self.ddpm.alphas_cumprod.to(self.device)
        return alpha_cumprod[t]

    def training_step(self, train_batch):
        self.ddpm.to(self.device)
        if self.type in ("scalars", "dims"):
            vars, idx = train_batch
        else:
            masks, vars, idx = train_batch 
        RP = vars[:,:,:200].float().to(self.device)
        scal = self._extract_scalars(vars)
        if self.type not in ("scalars", "dims"):
            masks = [masks[:, 0].to(self.device), masks[:, 1].to(self.device)]
        noise = torch.randn(RP.shape).to(self.device).float()
        timesteps = torch.randint(1, self.num_timesteps, (RP.shape[0],)).to(self.device)
        noisy = self.ddpm.add_noise(RP, noise, timesteps)
        noisy_tm1 = self.ddpm.add_noise(RP, noise, timesteps-1)
        
        if self.type in ("scalars", "dims"):
            noise_pred = self.ddpm.reverse(noisy, timesteps, scal)
        else:
            noise_pred = self.ddpm.reverse(noisy, timesteps, [masks, scal])
        
        loss = F.mse_loss(noise_pred, noise)
        with torch.no_grad():
            if self.test_inf("train"):
                generated, RP = self(train_batch[:self.nb_id-1])
                RP, generated = plot_generated_true(RP, generated, scal[:self.bs].cpu().numpy(), self.save_path, self.min_rp, self.max_rp, "train", self.current_epoch)
                self.compute_loss("train", idx, generated)

        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch):
        with torch.no_grad():
            if self.type in ("scalars", "dims"):
                vars, idx = val_batch
            else:
                masks, vars, idx = val_batch             
            RP = vars[:, :, :200].float().to(self.device)
            scal = self._extract_scalars(vars)
            if self.type not in ("scalars", "dims"):
                masks = [masks[:, 0].to(self.device), masks[:, 1].to(self.device)]
            noise = torch.randn(RP.shape).to(self.device).float()
            timesteps = torch.randint(1, self.num_timesteps, (RP.shape[0],)).to(self.device)
            noisy = self.ddpm.add_noise(RP, noise, timesteps)
            noisy_tm1 = self.ddpm.add_noise(RP, noise, timesteps-1)

            if self.type in ("scalars", "dims"):
                noise_pred = self.ddpm.reverse(noisy, timesteps, scal)
            else:
                noise_pred = self.ddpm.reverse(noisy, timesteps, [masks, scal]) 

            loss = F.mse_loss(noise_pred, noise)

            if self.test_inf("val"):
                # if os.path.exists(self.save_path+"/val_roll/"+str(self.val_batch_count)+".png"):
                #     pass
                # else:
                #     self.plot_training_results(noisy, timesteps, embs, scal, RP)
                print("Validation inference")
                start = time.time()
                generated, RP = self(val_batch[:self.nb_id-1])
                print("Time to compute inference : {} seconds".format(np.round(time.time()-start, 2)))
                RP, generated = plot_generated_true(RP, generated, scal[:self.bs].cpu().numpy(), self.save_path, self.min_rp, self.max_rp, "val", self.current_epoch)
                self.compute_loss("val", idx, generated)
                
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)

if __name__=="__main__":
    model = DDPMlight1D2D(DDPM, MyTinyUNet, "cpu", 1000)
    img = [torch.randn(100, 3, 224, 224), torch.randn(100, 3, 224, 224)]
    vars = [torch.randn(100, 1, 200), torch.randn(100, 1), torch.randn(100, 1), torch.randn(100, 1)]
    model([img, vars])
