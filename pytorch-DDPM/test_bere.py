import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pathlib import Path

# === Paramètres ===
image_path = '/home/ids/bnghiem-23/Projet-IA-Telecom-Paris/Dataset/BPAEC/actin'
image_size = 128
batch_size = 16
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

# === Prétraitement ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),                 # [0,1]
    transforms.Normalize([0.5], [0.5])     # [-1,1]
])

# === Chargement du dataset ===
dataset = datasets.ImageFolder(image_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Récupérer un batch
images = next(iter(dataloader))[0]        # [B, C, H, W]
images_visu = (images + 1) / 2            # Re-normalise en [0,1] pour affichage

# === Affichage
grid = make_grid(images_visu, nrow=4)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("Aperçu du batch (prétraité)")
plt.savefig(save_path / "preview_input_batch.png")
plt.show()
