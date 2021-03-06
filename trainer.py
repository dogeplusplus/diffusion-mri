import torch
import imageio
import mlflow
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from pathlib import Path
from einops import rearrange
from torchmetrics import MeanMetric
from torch.cuda.amp import GradScaler, autocast
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize, Lambda

from model import Unet
from utils import tile_images, save_graph
from diffusion import DiffusionModel
from beta_schedule import linear_beta_schedule


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
    timesteps = 1000
    schedule_fn = linear_beta_schedule
    betas = schedule_fn(timesteps=timesteps)

    image_size = 32
    channels = 3
    batch_size = 128
    sample_every = 5
    num_samples = 3 ** 2

    epochs = 20
    display_every = 100
    render_size = 128
    dim_mults = (1, 2, 4,)
    accumulation_steps = 4
    loss_type = "l1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_parameters = dict(
        dim=image_size,
        channels=channels,
        dim_mults=dim_mults,
        use_convnext=False,
    )

    diffusion_parameters = dict(
        shape=(channels, image_size, image_size),
        timesteps=timesteps,
    )

    model = Unet(**model_parameters)
    model.to(device)

    diffusion_model = DiffusionModel(model, betas, **diffusion_parameters)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    preprocessing = Compose([
        ToTensor(),
        Resize((image_size, image_size)),
        Lambda(lambda t: (t * 2) - 1),
    ])
    scaler = GradScaler()

    postprocessing = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: torch.from_numpy(t)),
        Lambda(lambda t: F.interpolate(t, [render_size, render_size])),
        Lambda(lambda t: rearrange(t, "b c h w -> b h w c")),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])

    dataset = CIFAR10(root="datasets", transform=preprocessing)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # Log configuration
    mlflow.log_params(diffusion_parameters)
    mlflow.log_param("beta_schedule", schedule_fn.__name__)
    mlflow.log_param("loss_type", loss_type)
    mlflow.log_dict(model_parameters, "model_config.json")

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir, "betas.npy")
        np.save(temp_file, betas.numpy())
        mlflow.log_artifact(temp_file)

        sample_x, _ = next(iter(dataloader))
        sample_x = sample_x.to(device)
        sample_t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        graph_path = save_graph(model, sample_x, sample_t, temp_dir)
        mlflow.log_artifact(graph_path)

    step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=0)
        loss_metric = MeanMetric()
        for batch in pbar:
            step += 1

            with autocast():
                images, _ = batch
                images = images.to(device)

                t = torch.randint(0, timesteps, (batch_size,), device=device).long()
                loss = p_losses(diffusion_model, images, t, loss_type=loss_type)
                # TODO: figure out why the loss is infinity for the MRI dataset
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            optimizer.step()

            if (step+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_metric.update(loss.item())

            if step % display_every == 0:
                pbar.set_postfix(loss=loss_metric.compute().item())

        if epoch % sample_every == 0:
            samples = diffusion_model.p_sample_loop(num_samples)
            image_samples = np.stack([postprocessing(x) for x in samples])
            image_samples = tile_images(image_samples)

            with TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir, "animation.gif")
                imageio.mimsave(temp_path, image_samples, fps=30)
                mlflow.log_artifact(temp_path, "gifs")

            mlflow.log_image(image_samples[-1], f"sample_{epoch:05d}.png")
            mlflow.pytorch.log_state_dict(model.state_dict(), "model")

        mlflow.log_metric("loss", loss_metric.compute().item(), step=epoch)


if __name__ == "__main__":
    main()
