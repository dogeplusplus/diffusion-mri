import torch
import torch.nn.functional as F

from torch import optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    RandomHorizontalFlip,
)

from model import Unet
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

    image_size = 28
    channels = 1
    batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1,2,4,)
    )
    model.to(device)

    diffusion_model = DiffusionModel(model, betas)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = load_dataset("fashion_mnist")
    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    epochs = 5

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(
                diffusion_model,
                batch,
                t,
                loss_type="huber"
            )

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model.pt")
    torch.save(model.betas, "betas.pt")


if __name__ == "__main__":
    main()
