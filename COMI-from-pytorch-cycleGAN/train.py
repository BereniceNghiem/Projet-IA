import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

import os
from torchvision.models import vgg19

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')  # default=200
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/Leishmania/', help='root directory of the dataset')  # default='datasets/horse2zebra/'
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')  # default=100
parser.add_argument('--height', type=int, default=384, help='height of the crop')
parser.add_argument('--width', type=int, default=512, help='width of the crop')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--perceptual_weight', type=float, default=1, help='Weight for perceptual loss')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


vgg = vgg19(pretrained=True).features[:16].eval()  # conv4_1
for param in vgg.parameters():
    param.requires_grad = False
if opt.cuda:
    vgg.cuda()

# Lossess
lambda_gan = 1          #1
lambda_content = 5      #1
lambda_cycle = 10       #0.001
lambda_identity = 5

criterion_GAN = torch.nn.BCELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.99))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.99))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.99))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.height, opt.width)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.height, opt.width)
target_real = Variable(Tensor(opt.batchSize, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize, 1).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize((int(opt.height*1.12), int(opt.width*1.12)), Image.BICUBIC),
                transforms.RandomCrop((opt.height, opt.width)),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda img: img.convert("RGB")), #ajout gpt pour avoir les bonnes dim datasets Xiangjing
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)


# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

#writer 

psnr_A2B, ssim_A2B, pcc_A2B, std_A2B = [], [], [], []
psnr_B2A, ssim_B2A, pcc_B2A, std_B2A = [], [], [], []
epoch_losses = []

loss_gan_list = []
loss_content_list = []
loss_cycle_list = []
loss_identity_list = []

psnr_A2B_epoch, ssim_A2B_epoch, pcc_A2B_epoch, std_A2B_epoch = [], [], [], []
psnr_B2A_epoch, ssim_B2A_epoch, pcc_B2A_epoch, std_B2A_epoch = [], [], [], []

os.makedirs("output/metrics", exist_ok=True)  # sauvegarde des données

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()


        # === GAN loss ===
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(target_real))

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(target_real))
        
        loss_GAN = loss_GAN_A2B + loss_GAN_B2A


        #  === Perceptual loss ===
        def vgg_features(x):
            return vgg((x + 1.0) / 2.0)  # De-normalize to [0,1]

        features_fake_B = vgg_features(fake_B)
        features_real_B = vgg_features(real_B)
        loss_perceptual_A2B = criterion_identity(features_fake_B, features_real_B)

        features_fake_A = vgg_features(fake_A)
        features_real_A = vgg_features(real_A)
        loss_perceptual_B2A = criterion_identity(features_fake_A, features_real_A)

        loss_content = (loss_perceptual_A2B + loss_perceptual_B2A) * opt.perceptual_weight


        # === Cycle loss ===
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

        loss_cycle = loss_cycle_ABA + loss_cycle_BAB

        # === Identity loss ===
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)

        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)

        loss_identity = loss_identity_A + loss_identity_B

        # Total loss
        loss_G = lambda_gan * loss_GAN + lambda_content * loss_content + lambda_cycle * loss_cycle + lambda_identity * loss_identity
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(target_real))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(target_fake))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(target_real))
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(target_fake))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # matplotlib and save

        # # --- Calcul des métriques ---
        # with torch.no_grad():
        #     fake_B_eval = netG_A2B(real_A)
        #     fake_A_eval = netG_B2A(real_B)

        #     real_A_np = real_A.cpu().numpy()
        #     real_B_np = real_B.cpu().numpy()
        #     fake_A_np = fake_A_eval.cpu().numpy()
        #     fake_B_np = fake_B_eval.cpu().numpy()

        #     for b in range(real_A_np.shape[0]):
        #         # A→B
        #         rB = real_B_np[b].transpose(1, 2, 0)
        #         fB = fake_B_np[b].transpose(1, 2, 0)

        #         psnr_A2B_epoch.append(psnr(rB, fB, data_range=1))
        #         ssim_A2B_epoch.append(ssim(rB, fB, channel_axis=2, data_range=1))
        #         pcc_A2B_epoch.append(pearsonr(rB.flatten(), fB.flatten())[0])
        #         std_A2B_epoch.append(np.std(rB - fB))

        #         # B→A
        #         rA = real_A_np[b].transpose(1, 2, 0)
        #         fA = fake_A_np[b].transpose(1, 2, 0)

        #         psnr_B2A_epoch.append(psnr(rA, fA, data_range=1))
        #         ssim_B2A_epoch.append(ssim(rA, fA, channel_axis=2, data_range=1))
        #         pcc_B2A_epoch.append(pearsonr(rA.flatten(), fA.flatten())[0])
        #         std_B2A_epoch.append(np.std(rA - fA))

        # tensorboard on loss
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A), 'loss_G_content': loss_content,
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_G_identity': loss_identity, 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')

    epoch_losses.append(loss_G.item() + loss_D_A.item() + loss_D_B.item())

    loss_gan_list.append(loss_GAN.item())
    loss_content_list.append(loss_content.item())
    loss_cycle_list.append(loss_cycle.item())
    loss_identity_list.append(loss_identity.item())

    # Moyennes des métriques pour l'époque
    psnr_A2B.append(np.mean(psnr_A2B_epoch))
    ssim_A2B.append(np.mean(ssim_A2B_epoch))
    pcc_A2B.append(np.mean(pcc_A2B_epoch))
    std_A2B.append(np.mean(std_A2B_epoch))

    psnr_B2A.append(np.mean(psnr_B2A_epoch))
    ssim_B2A.append(np.mean(ssim_B2A_epoch))
    pcc_B2A.append(np.mean(pcc_B2A_epoch))
    std_B2A.append(np.mean(std_B2A_epoch))

    # Reset des accumulateurs pour l’époque suivante
    psnr_A2B_epoch.clear()
    ssim_A2B_epoch.clear()
    pcc_A2B_epoch.clear()
    std_A2B_epoch.clear()

    psnr_B2A_epoch.clear()
    ssim_B2A_epoch.clear()
    pcc_B2A_epoch.clear()
    std_B2A_epoch.clear()

    # Sauvegarde des métriques à chaque époque
    np.save("output/metrics/psnr_A2B.npy", np.array(psnr_A2B))
    np.save("output/metrics/ssim_A2B.npy", np.array(ssim_A2B))
    np.save("output/metrics/pcc_A2B.npy",  np.array(pcc_A2B))
    np.save("output/metrics/std_A2B.npy",  np.array(std_A2B))

    np.save("output/metrics/psnr_B2A.npy", np.array(psnr_B2A))
    np.save("output/metrics/ssim_B2A.npy", np.array(ssim_B2A))
    np.save("output/metrics/pcc_B2A.npy",  np.array(pcc_B2A))
    np.save("output/metrics/std_B2A.npy",  np.array(std_B2A))

    np.save("output/metrics/losses.npy", np.array(epoch_losses))

    np.save("output/metrics/loss_gan.npy", np.array(loss_gan_list))
    np.save("output/metrics/loss_content.npy", np.array(loss_content_list))
    np.save("output/metrics/loss_cycle.npy", np.array(loss_cycle_list))
    np.save("output/metrics/loss_identity.npy", np.array(loss_identity_list))

###################################
# # Courbe combinée des métriques de qualité d'image avec double axe
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Axe principal : PSNR (échelle de 0 à 20)
# ax1.plot(psnr_A2B, label="PSNR A→B", color='tab:blue')
# ax1.plot(psnr_B2A, label="PSNR B→A", color='tab:cyan')
# ax1.set_ylabel("PSNR", color='tab:blue')
# ax1.set_ylim(0, 20)
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Axe secondaire : SSIM, PCC, STD (échelle de 0 à 1)
# ax2 = ax1.twinx()
# ax2.plot(ssim_A2B, label="SSIM A→B", color='tab:orange')
# ax2.plot(ssim_B2A, label="SSIM B→A", color='tab:red')
# ax2.plot(pcc_A2B, label="PCC A→B", color='tab:green')
# ax2.plot(pcc_B2A, label="PCC B→A", color='tab:olive')
# ax2.plot(std_A2B, label="STD A→B", color='tab:purple')
# ax2.plot(std_B2A, label="STD B→A", color='tab:pink')
# ax2.set_ylabel("SSIM / PCC / STD", color='tab:orange')
# ax2.set_ylim(0, 1)
# ax2.tick_params(axis='y', labelcolor='tab:orange')

# # Titre, axe x, légende combinée
# plt.title("Évolution des métriques de qualité (PSNR, SSIM, PCC, STD)")
# ax1.set_xlabel("Époque")

# # Combine les légendes des deux axes
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

# plt.grid()
# plt.tight_layout()
# plt.savefig("output/metrics/kpi_plot.png")
# plt.close()


# Courbes des composantes pondérées de la loss
plt.figure(figsize=(10, 6))
plt.plot([lambda_gan * l for l in loss_gan_list], label="Adversarial Loss (λ₁×)")
plt.plot([lambda_content * l for l in loss_content_list], label="Content Loss (λ₂×)")
plt.plot([lambda_cycle * l for l in loss_cycle_list], label="Cycle Consistency Loss (λ₃×)")
plt.plot([lambda_identity * l for l in loss_identity_list], label="Identity Loss (λ₄×)")
plt.plot(epoch_losses, label="Total Loss", linestyle='--')
plt.title("Évolution des composantes pondérées de la loss")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("output/metrics/loss_components.png")
plt.close()

# ================================
# ÉVALUATION A→B (images alignées)
# ================================
def denorm(tensor):
    return tensor * 0.5 + 0.5

# Dataset aligné (pas d'augmentation, pas de crop)
eval_transforms = [ 
    transforms.Resize((opt.height, opt.width), Image.BICUBIC),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
eval_dataloader = DataLoader(
    ImageDataset(opt.dataroot, transforms_=eval_transforms, unaligned=False),
    batch_size=1, shuffle=False, num_workers=opt.n_cpu
)

# Charge les poids
netG_A2B.load_state_dict(torch.load("output/netG_A2B.pth"))
netG_A2B.eval()
if opt.cuda:
    netG_A2B.cuda()

# Évaluation
psnr_list, ssim_list, pcc_list = [], [], []

with torch.no_grad():
    for i, batch in enumerate(eval_dataloader):
        real_A = batch["A"].cuda() if opt.cuda else batch["A"]
        real_B = batch["B"].cuda() if opt.cuda else batch["B"]

        fake_B = netG_A2B(real_A)

        real_B_np = denorm(real_B).cpu().numpy()[0].transpose(1, 2, 0)
        fake_B_np = denorm(fake_B).cpu().numpy()[0].transpose(1, 2, 0)

        psnr_val = psnr(real_B_np, fake_B_np, data_range=1.0)
        ssim_val = ssim(real_B_np, fake_B_np, data_range=1.0, channel_axis=2)
        pcc_val = pearsonr(real_B_np.flatten(), fake_B_np.flatten())[0]

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        pcc_list.append(pcc_val)

print("\n=== ÉVALUATION A→B ===")
print(f"PSNR : {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f}")
print(f"SSIM : {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
print(f"PCC  : {np.mean(pcc_list):.4f} ± {np.std(pcc_list):.4f}")