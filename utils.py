import torch
import numpy as np
import torch.nn as nn

from torchviz import make_dot
from einops import rearrange


def tile_images(images: np.ndarray) -> np.ndarray:
    batch = images.shape[0]
    num_samples = int(np.sqrt(batch))

    assert num_samples ** 2 == batch, "Batch size must be square"
    image_tiles = rearrange(
        images,
        "(y x) t h w c -> t (y h) (x w) c",
        y=num_samples,
        x=num_samples,
    )

    return image_tiles


def save_graph(model: nn.Module, x: torch.Tensor, t: torch.Tensor, save_dir: str) -> str:
    graph = make_dot(
        model(x, t),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    ext = "jpg"
    graph.format = ext
    dest = f"{save_dir}/graph"
    graph.render(dest)
    return f"{dest}.{ext}"
