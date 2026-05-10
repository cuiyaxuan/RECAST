from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class PipelineConfig:
    seed: int = 0
    latent_dim: int = 64
    batch_size: int = 256
    vae_epochs: int = 30
    proj_epochs: int = 200
    hvg_top_genes: int = 3000
    lr_vae: float = 1e-3
    lr_proj: float = 1e-3
    kl_weight: float = 1e-4
    use_deterministic: bool = True

    def make_output_dir(self, output_dir: Union[str, Path]) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out