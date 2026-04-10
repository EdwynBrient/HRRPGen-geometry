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
except ImportError:  # pragma: no cover
    from ema import EMA
    from utils import *
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import time

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

        out = self.block_2(out)
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

class EmbedBlock(nn.Module):
    """
    abstract class
    """
    def forward(self, x, cemb):
        """
        abstract method
        """

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

class EmbedBlockUncond(nn.Module):
    """
    abstract class
    """
    def forward(self, x):
        """
        abstract method
        """

class EmbedSequential(nn.Sequential, EmbedBlock, EmbedBlockUncond):
    def forward(self, x:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, cemb)
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

    def forward(self, x:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.c1emb_proj(cemb[0])[:, :, None]
        if self.dbcond is not None:
            latent += self.c2emb_proj(cemb[1])[:, :, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
    
class ResBlockUncond(EmbedBlockUncond):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, dropout:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dropout = dropout

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size = 3, padding = 1),
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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
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
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        output = self.layer(x)
        return output

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.mod_ch = config["mod_ch"]
        self.ch_mul = config["ch_mul"]
        self.num_res_blocks = config["num_res"]
        self.conditionned = config["conditionned"]
        self.dropout = config["dropout"]
        self.cond = self.conditionned["bool"]
        if self.cond:
            self.type = self.conditionned["type"]
            self.use_dims = self.conditionned.get("dims", True)
            self.use_va = self.conditionned.get("va", True)
        assert 25*(2**(len(self.ch_mul)-1)) == 200, "The final length of generation must be 200"
        cdim = self.mod_ch * 4

        if self.cond:    
            if self.type in ("scalars", "both", "dims"):
                scalar_inputs = (1 if self.use_va else 0) + (2 if self.use_dims else 0)
                if scalar_inputs == 0:
                    raise ValueError("At least one scalar (va or dimensions) must be enabled when conditioning on scalars.")
                self.scal_emb_layer = EmbedLinear(scalar_inputs*self.mod_ch, cdim)
            if self.type == "masks" or self.type=="both":
                self.make_masks_emb = FullMask(config, cdim)
                self.mask_emb_layer = EmbedLinear(784, cdim)

        dbcond = None
        if self.type=="both":
            dbcond = 1   

        now_ch = self.ch_mul[0] * self.mod_ch
        self.conv_in = nn.Conv1d(1, now_ch, 3, 1, 1)
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul)):
            nxt_ch = mul * self.mod_ch
            for j in range(self.num_res_blocks):
                if self.cond:
                    layers = [
                        ResBlock(now_ch, nxt_ch, cdim, cdim, self.dropout, dbcond),
                    ]
                else:
                    layers = [
                        ResBlockUncond(now_ch, nxt_ch, self.dropout),
                    ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks-1:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
        self.convout = nn.Conv1d(self.ch_mul[-1] * self.mod_ch, 1, 3, 1, 1)

    def forward(self, z, conds):
        if len(conds) == 2:
            mask, scal = conds
        else:
            scal = conds
        n = z.shape[0]
        scal = torch.concat([timestep_embedding(scal[:, i], self.mod_ch, 10) for i in range(scal.shape[1])], dim=1)
        if self.cond:
            if self.type in ("scalars", "both", "dims"):
                scal = self.scal_emb_layer(scal)
            if self.type == "masks" or self.type=="both":
                if self.type == "both":
                    mask1, mask2 = self.make_masks_emb(mask[0], scal).reshape(n, -1), self.make_masks_emb(mask[1], scal).reshape(n, -1)
                masks = torch.concat([mask1, mask2], dim=1)
                masks = self.mask_emb_layer(masks)
        if self.cond:
            if self.type == "both":
                cemb = [masks, scal]
            elif self.type in ("scalars", "dims"):
                cemb = [scal]
            else:
                cemb = [masks]
        x = self.conv_in(z)
        for block in self.upblocks:
            if self.cond:
                x = block(x, cemb)
            else:
                x = block(x, 0)
        return self.convout(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.featureExtractor = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 5, 2, 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, 2, 2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, 5, 1, 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        self.classif = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 25, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, input):
        emb = self.featureExtractor(input)
        emb = self.classif(emb)
        return emb

class GANlight(pl.LightningModule):
    def __init__(self, gen, disc, config, dataset, validation_indices, save_path, minmax):
        super().__init__()
        self.config = config
        self.generator = gen(config)
        self.discriminator = disc()
        self.dataset = dataset
        self.df = dataset.df
        self.df_va = dataset.old_va
        self.val_idx = validation_indices
        self.bs=30
        self.min_rp, self.max_rp = minmax
        self.inf_every = config["inf_every_n_epoch"]
        self.cond = config["conditionned"]["bool"]
        self.type = self.config["conditionned"]["type"]
        self.loss = self.config["loss"]
        self.save_path = save_path
        self.automatic_optimization = False
        self._last_infer_bucket = {"train": -1, "val": -1}  # step-based inference
        self._last_infer_epoch = {"train": -1, "val": -1}  # epoch-based inference
        self.use_va = self.config["conditionned"].get("va", True)
        self.use_dims = self.config["conditionned"].get("dims", True)

    def forward(self, inp, conds):
        return self.generator(inp, conds)

    def _extract_scalars(self, vars: torch.Tensor) -> torch.Tensor:
        scalars = []
        if self.use_va:
            scalars.append(vars[:, :, -3])
        if self.use_dims:
            scalars.extend([vars[:, :, -2], vars[:, :, -1]])
        if len(scalars) == 0:
            raise ValueError("No scalar features selected: enable at least one of conditionned.va or conditionned.dims.")
        return torch.concat(scalars, dim=1).float()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.config["lr"])
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.config["lr"])
        return [opt_g, opt_d], []

    def compute_loss(self, mode, idx, generated, sel=None):
        if sel is not None:
            t_loss = [top_loss(self.df, idx[i].cpu().item(), generated[i], self.min_rp, self.max_rp, 0.04) for i in sel.tolist()]
        else:
            t_loss = [top_loss(self.df, idx[i].cpu().item(), generated[i], self.min_rp, self.max_rp, 0.04) for i in range(generated.shape[0])]
        tpsnr, tcos_f, tmse_f, lpf, rcs_f = zip(*t_loss)  # chaque lpf est (L,)
        tpsnr  = torch.tensor(tpsnr,  dtype=torch.float32)
        tcos_f = torch.tensor(tcos_f, dtype=torch.float32)
        tmse_f = torch.tensor(tmse_f, dtype=torch.float32)
        lpf    = torch.tensor(lpf,    dtype=torch.float32)  # (B, L) -> OK
        rcs_f  = torch.tensor(rcs_f,  dtype=torch.float32)
        if mode != "test":
            self.log(f"top_psnr_{mode}", tpsnr.mean(), prog_bar=True)
            self.log(f"top_cosf_{mode}", tcos_f.mean(), prog_bar=True)
            self.log(f"top_msef_{mode}", tmse_f.mean(), prog_bar=True)
            self.log(f"top_rcs_{mode}", rcs_f.mean(), prog_bar=True)
        return tpsnr, tcos_f, tmse_f, lpf, rcs_f

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
    
    def training_step(self, train_batch):
        opt_g, opt_d = self.optimizers()
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.05, 0.05)
        if self.type in ("scalars", "dims"):
            vars, idx = train_batch
        else:
            masks, vars, idx = train_batch     
        RP = vars[:,:,:200].float().to(self.device)
        scal = self._extract_scalars(vars)
        if self.type not in ("scalars", "dims"):
            masks = [masks[:, 0].to(self.device), masks[:, 1].to(self.device)]
        z = torch.randn(RP.shape[0], 1, 25).to(self.device)

        # train generator
        gen = self(z, [masks, scal]) if self.type not in ("scalars", "dims") else self(z, scal)
        # adversarial loss
        self.toggle_optimizer(opt_g)
        adv_loss = F.softplus(-self.discriminator(gen)).mean()

        mse = F.mse_loss(gen, RP)
        mse_loss = mse

        # total loss
        g_loss = adv_loss + 100.*mse_loss
        self.log("g_loss", adv_loss)
        self.log("train_mse", mse_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # train discriminator
        # Valid identification
        self.toggle_optimizer(opt_d)
        real_loss = F.softplus(-self.discriminator(RP)).mean()

        # sample in latent space
        fake_RP = self(z, [masks, scal]) if self.type not in ("scalars", "dims") else self(z, scal)

        # Fake identification
        fake_loss = F.softplus(self.discriminator(fake_RP.detach())).mean()

        # discriminator accuracy (only for logs)
        with torch.no_grad():
            d_accuracy_fake = 1-torch.round(torch.sigmoid(self.discriminator(fake_RP))).mean()
            d_accuracy_real = (torch.round(torch.sigmoid(self.discriminator(RP)))).mean()
            d_acc = (d_accuracy_fake + d_accuracy_real) / 2

        # Average of these loss
        d_loss = real_loss + fake_loss
        self.log("d_loss", d_loss)
        self.log("d_acc", d_acc, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        with torch.no_grad():
            if self.test_inf("train"):
                RP, generated = plot_generated_true(RP[:self.bs], fake_RP[:self.bs].cpu().numpy(), scal[:self.bs].cpu().numpy(), self.save_path, self.min_rp, self.max_rp, "train", self.current_epoch)
                self.compute_loss("train", idx, generated)

    def validation_step(self, val_batch, val_idx):
        with torch.no_grad():
            if self.type in ("scalars", "dims"):
                vars, idx = val_batch
            else:
                masks, vars, idx = val_batch         
            RP = vars[:,:,:200].float().to(self.device)
            scal = self._extract_scalars(vars)
            if self.type not in ("scalars", "dims"):
                masks = [masks[:, 0].to(self.device), masks[:, 1].to(self.device)]
            z = torch.randn(RP.shape[0], 1, 25).to(self.device)
            generated = self(z, [masks, scal]) if self.type not in ("scalars", "dims") else self(z, scal)
            mse = F.mse_loss(generated, RP)
            mse_loss = mse
            d_accuracy_fake = 1-torch.round(torch.sigmoid(self.discriminator(generated))).mean()
            d_accuracy_real = (torch.round(torch.sigmoid(self.discriminator(RP)))).mean()
            d_acc = (d_accuracy_fake + d_accuracy_real) / 2        
            self.log("val_mse", mse_loss, prog_bar=True, sync_dist=True)
            self.log("val_D_acc", d_acc, prog_bar=True, sync_dist=True)
            gen = unnormalize_hrrp(generated.cpu().numpy(), self.min_rp, self.max_rp)
            B = gen.shape[0]
            take = min(B, self.config.get("val_metrics_max_samples", 1))
            sel = torch.arange(take, device=self.device)
            self.compute_loss("val", idx, gen, sel)
            if self.test_inf("val"):
                RP, generated = plot_generated_true(RP[:self.bs], generated[:self.bs].cpu(), scal[:self.bs].cpu().numpy(), self.save_path, self.min_rp, self.max_rp, "val", self.current_epoch)
