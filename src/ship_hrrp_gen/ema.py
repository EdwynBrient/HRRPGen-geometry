from typing import Any

import pytorch_lightning as pl
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        # Store shadow parameters on CPU
        self.shadow = {
            name: param.clone().detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Move param to CPU for update
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data
                    + (1.0 - self.decay) * param.data.cpu()
                )

    def apply_shadow(self):
        """Apply EMA weights to the model, moving them to the correct device."""
        # Keep backup on CPU to avoid doubling GPU memory footprint
        self.backup = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.shadow[name].to(param.device).data

    def restore(self):
        """Restore the original model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.backup[name].to(param.device).data
