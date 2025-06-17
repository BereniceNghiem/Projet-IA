from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import torchvision
from torchvision.transforms import ToPILImage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import numpy as np
import os

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False          # car on n'a pas de partition A100
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/home/ids/bnghiem-23/Projet-IA-Telecom-Paris/Dataset/BPAEC/actin',
    train_batch_size = 32,
    train_lr = 8e-5,
    # train_num_steps = 700000,         # total training steps
    train_num_steps = 250,       # à augmenetr jusqu'à 10000
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    # calculate_fid = True,              # whether to calculate fid during training
    calculate_fid = False,
    #num_fid_samples=1000,
    num_samples=64,
    save_and_sample_every=50        
)

trainer.train()