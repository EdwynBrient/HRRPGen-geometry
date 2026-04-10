import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from transformers import AutoModelForImageClassification
import pytorch_lightning as pl
from math import pi
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import torchvision
import copy
import os
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
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim:int, cdim:int, dropout:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
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
        
        self.cemb_proj = nn.Sequential(
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
        latent += self.cemb_proj(cemb)[:, :, None]
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

class EmbedLinear(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, ):
        super().__init__()
        self.layer1 = nn.Linear(in_ch, out_ch)
        self.silu = nn.SiLU()
        self.layer2 = nn.Linear(out_ch, out_ch)
        self.layernorm = nn.LayerNorm(out_ch)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.layer2(self.silu(self.layer1(x))))

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

class Downsample(nn.Module):
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
        self.class_free = config["class_free"]["bool"]
        self.dropout = config["dropout"]
        if self.class_free:
            self.guidance = config["class_free"]["guidance"]
                           
        tdim = self.mod_ch * 4
        self.temb_layer = EmbedLinear(self.mod_ch, tdim)

        if self.cond:    
            if self.conditionned["dims"] and self.conditionned["va"]:
                self.scal_emb_layer = EmbedLinear(3*self.mod_ch, tdim)
            elif self.conditionned["dims"]:
                self.scal_emb_layer = EmbedLinear(2*self.mod_ch, tdim)
            elif self.conditionned["va"]:
                self.scal_emb_layer = EmbedLinear(self.mod_ch, tdim)
            cdim = tdim

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
                        ResBlock(now_ch, nxt_ch, tdim, cdim, self.dropout),
                    ]
                else:
                    layers = [
                        ResBlockUncond(now_ch, nxt_ch, tdim, self.dropout),
                    ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(Downsample(now_ch, now_ch)))
                chs.append(now_ch)
        if self.cond:
            self.middleblocks = EmbedSequential(
                ResBlock(now_ch, now_ch, tdim, cdim, self.dropout),
                ResBlock(now_ch, now_ch, tdim, cdim, self.dropout)
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
                        ResBlock(now_ch+chs.pop(), nxt_ch, tdim, cdim, self.dropout),
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
        va = c[:, 0].unsqueeze(1)
        dims = c[:, 1:]
        if self.conditionned["dims"] and self.conditionned["va"]:
            vars = torch.concat([va, dims], dim=1)
        elif self.conditionned["dims"]:
            vars = dims
        elif self.conditionned["va"]:
            vars = va

        vars = torch.concat([timestep_embedding(vars[:, i], self.mod_ch, 10) for i in range(vars.shape[1])], dim=1)
        if self.cond:
            vars = self.scal_emb_layer(vars).reshape(n, -1)
            if self.training:
                vars = vars + torch.randn_like(vars) * self.conditionned["noise"]
            if self.class_free:
                sample_guidance = np.random.uniform(size=n)
                vars[sample_guidance > self.guidance] = 0
                
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            if self.cond:
                h = block(h, temb, vars)
            else:
                h = block(h, temb, 0)
            hs.append(h)
        if self.cond:
            h = self.middleblocks(h, temb, vars)
        else:
            h = self.middleblocks(h, temb, 0)
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim = 1)
            if self.cond:
                h = block(h, temb, vars)
            else:
                h = block(h, temb, 0)
        h = h.type(self.dtype)
        return self.out(h)
    
    def two_linear(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))


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

    def step(self, model_output, timestep, sample):
        """
        Perform one reverse diffusion step using dynamically computed values.
        Args:
            model_output (torch.Tensor): Predicted noise by the model.
            timestep (int): Current timestep.
            sample (torch.Tensor): Current noisy sample.
        Returns:
            torch.Tensor: Predicted sample at the previous timestep.
        """
        # Get dynamic betas and alphas
        betas = self.var_scheduler().to(sample.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(sample.device)

        # Gather values for the given timestep
        alpha_t = alphas[timestep]
        alpha_hat_t = alphas_cumprod[timestep]

        # Compute coefficients
        coef_eps_t = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
        coef_first = 1 / torch.sqrt(alpha_t)
        coef_eps_t = coef_eps_t.view(-1, 1, 1).to(sample.device)
        coef_first = coef_first.view(-1, 1, 1).to(sample.device)

        # Predict previous sample
        pred_prev_sample = coef_first * (sample - coef_eps_t * model_output)

        # Add noise for variance
        variance = 0
        if timestep > 0:
            noise = torch.randn_like(model_output).to(sample.device)
            variance = torch.sqrt(betas[timestep]) * noise

        return pred_prev_sample + variance

class DDPMlight1D2D(pl.LightningModule):
    def __init__(self, ddpm, config, save_path,minmax):
        super().__init__()
        self.config = config
        self.automatic_optimization = True
        self._compiler_ctx = None
        self.train_inference_time = False
        self.val_inference_time = False
        self.min_rp, self.max_rp = minmax
        self.ddpm = ddpm
        self.ema_bool = config["ema"]
        self.num_timesteps = config["num_timesteps"]
        self.inf_every = config["inf_every_n_epoch"]
        scheduler = load_scheduler_from_config(config)
        self.save_path = save_path
        if self.config["gradient_monitoring"]:
            self.gradient_norms = []
            self.gradient_max_val = []
        self.val_batch_count=0
        self.train_batch_count=0
        if self.config["scheduler"]["name"]=="learnable":
            self.lambda_smooth = torch.Tensor([config["scheduler"]["lambda_smooth"]])
        self.scheduler = scheduler
        if self.ema_bool:
            self.ema = EMA(self.ddpm.network, decay=config["ema_decay"])

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

    # def plot_training_results(self, noisy, timesteps, embs, vars, RP):
    #     os.makedirs(self.save_path + "/val_roll/", exist_ok=True)
    #     noisy_to_plot = unnormalize_hrrp(noisy.detach().cpu().numpy()[:8], 0., 216.)
    #     timesteps = timesteps.to(self.device)
    #     timesteps_plot = timesteps.cpu().numpy()[:8]
    #     RP = unnormalize_hrrp(RP[:8].detach().cpu().numpy(), 0., 216.)
    #     embs = [embs[0][:8], embs[1][:8]]
    #     x0 = self.roll_x0_from_xt(noisy[:8], timesteps[:8], [embs, vars[:8]])
    #     x0 = unnormalize_hrrp(x0.detach().cpu().numpy(), 0., 216.)
    #     to_plot = np.zeros((4, 6, 200))
    #     little_title = [[] for _ in range(4)]
    #     for i in range(4):
    #         for j in range(2):
    #             to_plot[i, 3*j, :] = noisy_to_plot[2*i+j]
    #             to_plot[i, 3*j+1, :] = x0[2*i+j]
    #             to_plot[i, 3*j+2, :] = RP[2*i+j]
    #             little_title[i].append("Noisy t = {}".format(timesteps_plot[2*i+j]))
    #             little_title[i].append("Rolled")
    #             little_title[i].append("True")
    #     plot_HRRP(to_plot, "Rolled batch number {}".format(self.val_batch_count),
    #               little_title,
    #               self.save_path + "/val_roll/" + str(self.val_batch_count) + ".png", True)
    #     plt.close('all')

    def forward(self, sample_batch):
        with torch.no_grad():
            bs=24
            self.ddpm.to(self.device)
            frames = []
            timesteps = list(range(1, self.ddpm.num_timesteps))[::-1]
            vars = sample_batch
            vars = vars[:bs]
            RP = vars[:, 0, :200].float()

            sample = torch.randn(bs, 1, 200).float().to(self.device)
            vars = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float()
            
            for i, t in tqdm(enumerate(timesteps)):
                time_tensor = (torch.ones(RP.shape[0], 1) * t).long().to(self.device).squeeze()
                residual = self.ddpm.reverse(sample, time_tensor, vars)
                sample = self.ddpm.step(residual, time_tensor[0], sample)
                print("Extreme values at time {} : ".format(str(t)), torch.max(sample), torch.min(sample))

            for i in range(RP.shape[0]):
                frames.append(sample[i].detach().cpu().numpy())
        return frames, RP

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.ddpm.network.parameters(), lr=self.config["lr"])
        if self.config["cosine_annealing"] and self.config["gradient_clip_val"] != 0.0:
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

    def training_step(self, train_batch):
        self.ddpm.to(self.device)
        vars = train_batch 
        RP = vars[:,:,:200].float().to(self.device)
        vars = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float()
        noise = torch.randn(RP.shape).to(self.device).float()

        timesteps = torch.randint(0, self.num_timesteps, (vars.shape[0],)).to(self.device)
        noisy = self.ddpm.add_noise(RP, noise, timesteps)
        noise_pred = self.ddpm.reverse(noisy, timesteps, vars)
        if self.config["loss"]=="mse":
            loss = F.mse_loss(noise_pred, noise)
        elif self.config["loss"]=="wmse":
            loss = weighted_mse_loss_threshold(noise_pred, noise, weight_factor=2.0, threshold_ratio=0.3)
        else:
            raise ValueError("Loss must be either 'mse' or 'wmse'")
        if self.config["gradient_monitoring"]:
            self.gradient_norms, self.gradient_max_val = update_grad_lists(self.ddpm.network, self.gradient_norms, self.gradient_max_val)

        if self.current_epoch%self.inf_every!=0 and not self.train_inference_time:
            self.train_inference_time=True
        if self.current_epoch%self.inf_every==0 and self.train_inference_time:
            generated, RP = self(train_batch)
            plot_generated_true(RP, generated, self.save_path, self.min_rp, self.max_rp, "train", self.current_epoch)
            self.train_inference_time=False
        self.log("loss", loss, prog_bar=True)
        self.train_batch_count+=1
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_bool:
            self.ema.update()
        else:
            pass

    def on_validation_epoch_start(self):
        if self.ema_bool and self.train_batch_count>0:
            self.ema.apply_shadow()
        else:
            pass

    def on_validation_epoch_end(self):
        if self.ema_bool and self.train_batch_count>0:
            self.ema.restore()
        else:
            pass
    
    def on_validation_batch_end(self, *args, **kwargs):
        self.val_batch_count+=1
        if self.config["gradient_monitoring"] and self.val_batch_count!=1:
            mean_norm_grad = torch.Tensor([np.mean(self.gradient_norms)])
            max_norm_grad = torch.Tensor([np.max(self.gradient_max_val)])
            self.log("mean_norm_grad", mean_norm_grad, prog_bar=False)
            self.log("max_norm_grad", max_norm_grad, prog_bar=False)

    def validation_step(self, val_batch):
        with torch.no_grad():
            vars = val_batch
            RP = vars[:, :, :200].float().to(self.device)
            scal = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float()
            noise = torch.randn(RP.shape).to(self.device).float()
            timesteps = torch.randint(0, self.num_timesteps, (vars.shape[0],)).to(self.device)
            noisy = self.ddpm.add_noise(RP, noise, timesteps)
            noise_pred = self.ddpm.reverse(noisy, timesteps, scal)
            if self.current_epoch%self.inf_every!=0 and not self.val_inference_time:
                self.val_inference_time=True
            if self.current_epoch%self.inf_every==0 and self.val_inference_time:
                # if os.path.exists(self.save_path+"/val_roll/"+str(self.val_batch_count)+".png"):
                #     pass
                # else:
                #     self.plot_training_results(noisy, timesteps, embs, scal, RP)
                print("Validation inference")
                start = time.time()
                generated, RP = self(val_batch)
                print("Time to compute inference : {} seconds".format(np.round(time.time()-start, 2)))
                plot_generated_true(RP, generated, self.save_path, self.min_rp, self.max_rp, "val", self.current_epoch)
                self.val_inference_time=False
                
            loss = F.mse_loss(noise_pred, noise)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)


if __name__=="__main__":
    model = DDPMlight1D2D(DDPM, MyTinyUNet, "cpu", 1000)
    img = [torch.randn(100, 3, 224, 224), torch.randn(100, 3, 224, 224)]
    vars = [torch.randn(100, 1, 200), torch.randn(100, 1), torch.randn(100, 1), torch.randn(100, 1)]
    model([img, vars])
