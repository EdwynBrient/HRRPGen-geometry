"""
ship_hrrp_gen
------

Minimal package wrapper for the diffusion and GAN training code.
The modules are intentionally lightweight so they can be imported
from notebooks or scripts without touching the original folder.
"""

__all__ = [
    "ddpm",
    "gan",
    "utils",
    "dataset",
    "schedulers",
]
