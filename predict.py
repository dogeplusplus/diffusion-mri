import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Unet
from diffusion import DiffusionModel
from beta_schedule import linear_beta_schedule


def main():
    image_size = 28
    channels = 1
    dim_mults = (1,2,4,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=image_size, channels=channels, dim_mults=dim_mults)
    model.to(device)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    timesteps = 200
    betas = linear_beta_schedule(timesteps)
    diffusion = DiffusionModel(model, betas)

    batch_size = 32

    samples = diffusion.p_sample_loop((batch_size, channels, image_size, image_size), timesteps)
    idx = np.random.randint(0, batch_size)
    ani = generate_animation([x[idx].reshape(image_size, image_size, channels) for x in samples])
    plt.show()


if __name__ == "__main__":
    main()
