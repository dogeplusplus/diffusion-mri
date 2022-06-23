import torch
import mlflow
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from torch import optim
from pathlib import Path
from torchvision import transforms
from tempfile import TemporaryDirectory
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader

from model import Unet
from diffusion import DiffusionModel
from beta_schedule import linear_beta_schedule


def generate_animation(images):
    fig = plt.figure()
    imgs = []
    for img in images:
        im = plt.imshow(img, cmap="gray", animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(
        fig,
        imgs,
        interval=10,
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
    timesteps = 200
    betas = linear_beta_schedule(timesteps=timesteps)

    image_size = 64
    channels = 1
    batch_size = 8
    sample_every = 5

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
    images = [img for img in images if 60 <= int(str(img)[-8:-4]) <= 70]

    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
    ])
    dataset = Flowers102(root="datasets", download=True,
                         transform=preprocessing)

    # dataset = MRIDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    epochs = 100

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=0)
        for batch in pbar:
            images, _ = batch
            images = images.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, timesteps, (batch_size,),
                              device=device).long()

            loss = p_losses(
                diffusion_model,
                images,
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
            idx = np.random.randint(batch_size)
            animation = generate_animation(
                [x[idx].reshape(image_size, image_size, channels) for x in samples])

            with TemporaryDirectory() as tmpdir:
                temp_path = f"{tmpdir}/epoch_{epoch}.gif"
                animation.save(temp_path)
                mlflow.log_artifact(temp_path, "samples")

            mlflow.pytorch.log_state_dict(model.state_dict(), "model")


if __name__ == "__main__":
    main()
