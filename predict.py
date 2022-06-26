import json
import torch
import mlflow
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from einops import rearrange
from ast import literal_eval
from matplotlib import animation
from tempfile import TemporaryDirectory
from torchvision.transforms import Compose, Lambda

from model import Unet
from utils import tile_images
from diffusion import DiffusionModel


def generate_animation(images: np.ndarray) -> animation.Animation:
    fig = plt.figure()
    imgs = []
    for img in images:
        im = plt.imshow(img, cmap="gray", animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(
        fig,
        imgs,
        interval=1,
        blit=True,
        repeat=False,
        repeat_delay=3000,
    )
    plt.axis("off")
    plt.tight_layout()
    return animate


def load_run(client: mlflow.tracking.MlflowClient, run_id: str) -> nn.Module:
    run = client.get_run(run_id)
    diffusion_params = run.data.params
    diffusion_params = {k: literal_eval(v) for k, v in diffusion_params.items()}

    with TemporaryDirectory() as temp_dir:
        state_dict = client.download_artifacts(run_id, "model/state_dict.pth", temp_dir)
        config_path = client.download_artifacts(run_id, "model_config.json", temp_dir)

        with open(config_path, "r") as f:
            model_config = json.load(f)

        beta_path = client.download_artifacts(run_id, "betas.npy")
        betas = torch.from_numpy(np.load(beta_path))

        net = Unet(**model_config)
        net.load_state_dict(torch.load(state_dict))

    diffusion_model = DiffusionModel(net, betas, **diffusion_params)

    return diffusion_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inference script for diffusion model")
    parser.add_argument("--run", type=str, help="Run ID in mlflow")
    parser.add_argument("--samples", type=int, help="Number of samples to display", default=9)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    client = mlflow.tracking.MlflowClient()
    model = load_run(client, args.run)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = args.samples

    postprocessing = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: torch.from_numpy(t)),
        Lambda(lambda t: rearrange(t, "b c h w -> b h w c")),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])

    samples = model.p_sample_loop(batch_size)
    image_stack = np.stack([postprocessing(x) for x in samples])
    tiles = tile_images(image_stack)

    generate_animation(tiles)
    plt.show()


if __name__ == "__main__":
    main()
