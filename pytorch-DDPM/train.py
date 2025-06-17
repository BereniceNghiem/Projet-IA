from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from itertools import cycle

# === EMA helper ===
class EMA:
    def __init__(self, model, beta=0.995):
        self.model = model
        self.ema_model = Unet(
            dim = 128,
            dim_mults = (1, 2, 2, 4),
            #dim=64,
            #dim_mults=(1, 2, 4, 8),
            flash_attn=False
        ).to(next(model.parameters()).device)
        self.beta = beta
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()

    def update(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.beta * ema_param.data + (1 - self.beta) * param.data

    def copy_to(self, target_model):
        target_model.load_state_dict(self.ema_model.state_dict())

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Entraînement sur : {device}")

# === Hyperparamètres ===
image_path = '/home/ids/bnghiem-23/Projet-IA-Telecom-Paris/pytorch-DDPM/Dataset/mitochondria'
image_size = 128
batch_size = 16  #32 pas adapté à P100 ou 3090
num_steps = 10000
save_every = 1000
lr = 8e-5

# === Préparation du dataset ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(image_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
dl_iter = cycle(dataloader)

# === Modèle ===
model = Unet(
    dim = 128,
    dim_mults = (1, 2, 2, 4),
    flash_attn=False
).to(device)

ema_helper = EMA(model, beta=0.995)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,
    sampling_timesteps=250  
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)

# === Fonction de métriques ===
def compute_metrics(originals, reconstructions):
    metrics = {'psnr': [], 'ssim': [], 'pcc': [], 'std': []}
    
    for i in range(originals.size(0)):
        # Dénormalisation à [0, 1]
        orig = (originals[i].detach().cpu() + 1) / 2
        recon = (reconstructions[i].detach().cpu() + 1) / 2
        
        # Conversion en numpy
        orig_np = orig.permute(1, 2, 0).numpy()  # CHW -> HWC
        recon_np = recon.permute(1, 2, 0).numpy()
        
        # Si images RGB, convertir en niveaux de gris avec pondération
        if orig_np.shape[2] == 3:
            # Pondération RGB standard pour niveaux de gris
            weights = np.array([0.299, 0.587, 0.114])
            orig_gray = np.dot(orig_np, weights)
            recon_gray = np.dot(recon_np, weights)
        else:
            orig_gray = orig_np[:, :, 0]  # Single channel
            recon_gray = recon_np[:, :, 0]
        
        # Calcul des métriques
        metrics['psnr'].append(psnr(orig_gray, recon_gray, data_range=1.0))  # data_range=1.0 car dans [0,1]
        metrics['ssim'].append(ssim(orig_gray, recon_gray, data_range=1.0))
        
        # Pour PCC, assurez-vous que les valeurs sont aplaties
        metrics['pcc'].append(np.corrcoef(orig_gray.flatten(), recon_gray.flatten())[0, 1])
        
        # Écart-type sur l'image reconstruite en niveaux de gris
        metrics['std'].append(np.std(recon_gray))

    return {k: np.mean(v) for k, v in metrics.items()}

# === Entraînement + métriques ===
# resume = results_folder / "checkpoint_latest.pt"
# start_step = 1
# if resume.exists():
#     checkpoint = torch.load(resume)
#     model.load_state_dict(checkpoint['model'])
#     ema_helper.ema_model.load_state_dict(checkpoint['ema'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_step = checkpoint['step'] + 1
#     print(f">>> Reprise depuis l'étape {start_step}")

metric_log = []
loss_log = []

for step in range(1, num_steps + 1):
    batch = next(dl_iter)[0].to(device)

    t = torch.randint(0, diffusion.num_timesteps, (batch.size(0),), device=device)
    loss = diffusion.p_losses(x_start=batch, t=t)
    loss.backward()

    print(f"[{step}/{num_steps}] Loss: {loss.item():.4f}")
    loss_log.append({'step': step, 'loss': loss.item()})

    del loss
    torch.cuda.empty_cache()
    optimizer.step()
    optimizer.zero_grad()

    ema_helper.update()

    if step % save_every == 0 or step==1:
        # Création d'un modèle temporaire pour échantillonnage
        ema_sample_model = Unet(
            dim = 128,
            dim_mults = (1, 2, 2, 4),
            flash_attn=False
        ).to(device)
        ema_helper.copy_to(ema_sample_model)

        # On remplace temporairement le modèle utilisé dans diffusion
        diffusion.model = ema_sample_model
        samples = diffusion.sample(batch_size=batch_size)
        reconstructions = diffusion.reconstruct(batch, t_fixed=500)
        diffusion.model = model  # on restaure le vrai modèle après

        save_image(samples, results_folder / f"sample-{step}.png", nrow=8)

        metrics = compute_metrics(batch, reconstructions)
        metrics["step"] = step
        metric_log.append(metrics)

        print(f"→ Metrics @ step {step}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, PCC={metrics['pcc']:.4f}, STD={metrics['std']:.2f}")

        # CSV
        with open(results_folder / "metrics_log.csv", "a") as f:
            f.write(f"{step},{metrics['psnr']},{metrics['ssim']},{metrics['pcc']},{metrics['std']}\n")

        # Sauvegarde du modèle
        torch.save(model.state_dict(), results_folder / f"model_step_{step}.pt")

        torch.save({
            'step': step,
            'model': model.state_dict(),
            'ema': ema_helper.ema_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, results_folder / 'checkpoint_latest.pt')

        # Nettoyage mémoire GPU
        torch.cuda.empty_cache()

# === Courbes ===
df = pd.DataFrame(metric_log).set_index("step")
df.plot(title="Évolution des métriques")
plt.xlabel("Étape")
plt.ylabel("Valeur")
plt.grid()
plt.savefig(results_folder / "metrics_plot.png")
plt.close()

df_loss = pd.DataFrame(loss_log).set_index("step")
df_loss.plot(title="Évolution de la loss")
plt.xlabel("Étape")
plt.ylabel("Loss")
plt.grid()
plt.savefig(results_folder / "loss_plot.png")
plt.close()
