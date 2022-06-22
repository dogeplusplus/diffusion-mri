import cv2
import torch
import mlflow
import typing as t
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from torch import optim

from einops import repeat
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    RandomHorizontalFlip,
    Resize,
)

from model import Unet
from diffusion import DiffusionModel
from beta_schedule import linear_beta_schedule


class GITract(Dataset):
    def __init__(self, images: t.List[Path], image_shape: t.Tuple[int, int] = (224, 224)):
        self.images = np.array(images)
        self.image_shape = image_shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]

        img = np.load(img_path)
        img = cv2.resize(img, self.image_shape)

        img = np.asarray(img, dtype=np.float32)
        img /= img.max()
        img = repeat(img, "h w -> c h w", c=3)


        return img


def generate_animation(images):
    fig = plt.figure()
    imgs = []
    for img in images:
        im = plt.imshow(img, cmap="gray", animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(fig, imgs, interval=25, blit=True, repeat=False)
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
        loss =  F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def main():
    transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])

    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

    # Beta schedule
    timesteps = 200
    betas = linear_beta_schedule(timesteps=timesteps)

    image_size = 224
    channels = 3
    batch_size = 4
    sample_every = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1,2,4,)
    )
    model.to(device)

    diffusion_model = DiffusionModel(model, betas)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    images = list(Path("../gi-tract/datasets/2d/images").rglob("*.npy"))

    dataset = GITract(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    epochs = 5

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(
                diffusion_model,
                batch,
                t,
                loss_type="huber"
            )

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        if epoch % sample_every == 0:
            samples = diffusion_model.p_sample_loop(
                shape=(batch_size, channels, image_size, image_size),
                timesteps=timesteps,
            )
            torch.save(samples, f"samples/epoch_{epoch}.pt")
            idx = np.random.randint(batch_size)
            animation = generate_animation([x[idx].reshape(image_size, image_size, channels) for x in samples])

            with TemporaryDirectory() as tmpdir:
                temp_path = f"{tmpdir}/epoch_{epoch}.gif"
                animation.save(temp_path)
                mlflow.log_artifact(temp_path)

            mlflow.pytorch.log_state_dict(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
