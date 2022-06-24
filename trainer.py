import torch
import imageio
import mlflow
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from torch import optim
from pathlib import Path
from einops import rearrange
from torchmetrics import MeanMetric
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize, Lambda

from beta_schedule import linear_beta_schedule

from diffusion import DiffusionModel
from model import Unet


def generate_animation(images):
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


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def p_losses(
    diffusion_model,
    x_start,
    t,
    noise=None,
    loss_type="l1"
):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = diffusion_model.q_sample(x_start, t, noise)
    predicted_noise = diffusion_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def main():
    # Beta schedule
    timesteps = 100
    betas = linear_beta_schedule(timesteps=timesteps)

    image_size = 32
    channels = 3
    batch_size = 128
    sample_every = 5

    epochs = 5
    display_every = 100
    render_size = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    diffusion_model = DiffusionModel(model, betas)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    images = list(Path("../gi-tract/datasets/2d/images").rglob("*.npy"))
    images = [img for img in images if 65 <= int(str(img)[-8:-4]) <= 70]

    preprocessing = Compose([
        ToTensor(),
        Resize((image_size, image_size)),
    ])
    dataset = CIFAR10(root="datasets", download=True, transform=preprocessing)

    postprocessing = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: torch.from_numpy(t)),
        Lambda(lambda t: F.interpolate(t, [render_size, render_size])),
        Lambda(lambda t: rearrange(t, "b c h w -> b h w c")),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])

    # dataset = MRIDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=0)
        loss_metric = MeanMetric()
        for batch in pbar:
            step += 1
            images, _ = batch
            images = images.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(
                diffusion_model,
                images,
                t,
                loss_type="huber"
            )

            loss.backward()
            optimizer.step()

            loss_metric.update(loss.item())

            if step % display_every == 0:
                pbar.set_postfix(loss=loss_metric.compute().item())

        if epoch % sample_every == 0:
            samples = diffusion_model.p_sample_loop(
                shape=(batch_size, channels, image_size, image_size),
                timesteps=timesteps,
            )
            idx = np.random.randint(batch_size)
            image_samples = postprocessing(samples[idx])
            # animation = generate_animation(image_samples)

            with TemporaryDirectory() as tmpdir:
                temp_path = f"{tmpdir}/epoch_{epoch:05d}.gif"
                imageio.mimsave(temp_path, image_samples, fps=30)
                mlflow.log_artifact(temp_path, "samples")

            mlflow.log_metric("loss", loss_metric.compute().item())
            mlflow.pytorch.log_state_dict(model.state_dict(), "model")


if __name__ == "__main__":
    main()
