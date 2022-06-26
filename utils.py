import numpy as np

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
