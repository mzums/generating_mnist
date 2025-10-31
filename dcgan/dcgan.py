from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 2
batch_size = 256
image_size = 28

nc = 1
nz = 100
ngf = 64
ndf = 28
num_epochs = 20
lr = 0.0002

beta1 = 0.5
ngpu = 1

dataset = dset.MNIST(root='', transform=transforms.ToTensor(), download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 3, 2, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)
    

x_data = np.arange(0.001,1,0.001)
sat_loss_data = np.log(1-x_data)
sat_loss_derivation_data = -1/(1-x_data)
non_sat_loss_data = -np.log(x_data)
non_sat_loss_derivation_data = -1/x_data

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x_data, sat_loss_data, 'r', label='Saturating G loss')
# ax.plot(x_data, sat_loss_derivation_data, 'r--', label='Derivation saturating G loss')
# ax.plot(x_data, non_sat_loss_data, 'b', label='non-saturating loss')
# ax.plot(x_data, non_sat_loss_derivation_data, 'b--', label='Derivation non-saturating G loss')
# ax.set_xlim([0, 1])
# ax.set_ylim([-10, 4])
# ax.grid(True, which='both')
# ax.axhline(y=0, color='k')
# ax.axvline(x=0, color='k')
# ax.set_title('Saturating and non-saturating loss function')
# plt.xlabel('D(G(z))')
# plt.ylabel('Loss / derivation of loss')
# ax.legend()
# plt.show()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, nz, 1, 1, device=device)


def training_loop(num_epochs=num_epochs, saturating=False):
    
    netG = Generator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)
    
    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)
    
    criterion = nn.BCEWithLogitsLoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    

    img_list = []
    G_losses = []
    G_grads_mean = []
    G_grads_std = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    print("epochs:", num_epochs)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            
            if saturating:
                label.fill_(fake_label)
            else:
                label.fill_(real_label)
            
            output = netD(fake).view(-1)
            
            if saturating:
                errG = -criterion(output, label)
            else:
                errG = criterion(output, label)
            
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            G_grad = [p.grad.view(-1).cpu().numpy() for p in list(netG.parameters()) if p.grad is not None]
            if G_grad:
              G_grads_mean.append(np.concatenate(G_grad).mean())
              G_grads_std.append(np.concatenate(G_grad).std())
            else:
              G_grads_mean.append(0)
              G_grads_std.append(0)

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    
    return G_losses, D_losses, G_grads_mean, G_grads_std, img_list


G_losses_sat, D_losses_sat, G_grads_mean_sat, G_grads_std_sat, img_list_sat = training_loop(saturating=True)

G_losses_nonsat, D_losses_nonsat, G_grads_mean_nonsat, G_grads_std_nonsat, img_list_nonsat = training_loop(saturating=False)

# plt.figure(figsize=(10,5))
# plt.title("Generator and discriminator loss")
# plt.plot(G_losses_sat,label="Saturating G loss", alpha=0.75)
# plt.plot(D_losses_sat,label="Saturating D loss", alpha=0.75)
# plt.plot(G_losses_nonsat,label="Non-saturating G loss", alpha=0.75)
# plt.plot(D_losses_nonsat,label="Non-saturating D loss", alpha=0.75)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
# 
# plt.figure(figsize=(10,5))
# plt.title("Generator gradient means")
# plt.plot(G_grads_mean_sat, label="Saturating G loss", alpha=0.75)
# plt.plot(G_grads_mean_nonsat, label="Non-saturating G loss", alpha=0.75)
# plt.xlabel("Iterations")
# plt.ylabel("Gradient mean")
# plt.legend()
# plt.show()
# 
# 
# plt.figure(figsize=(10,5))
# plt.title("Generator gradient standard deviations")
# plt.plot(G_grads_std_sat,label="Saturating G loss", alpha=0.75)
# plt.plot(G_grads_std_nonsat,label="Non-saturating G loss", alpha=0.75)
# plt.xlabel("Iterations")
# plt.ylabel("Gradient standard deviation")
# plt.legend()
# plt.show()
# 
# fig = plt.figure(figsize=(8,8))
# plt.title('Saturating G loss')
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list_sat]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# HTML(ani.to_jshtml())
# 
# fig = plt.figure(figsize=(8,8))
# plt.title('Non-saturating G loss')
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list_nonsat]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# HTML(ani.to_jshtml())

real_batch = next(iter(dataloader))

plt.figure(figsize=(15,15))

plt.subplot(1,3,1)
plt.axis("off")
plt.title("Real images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,3,2)
plt.axis("off")
plt.title("Fake images - saturating G loss")
plt.imshow(np.transpose(img_list_sat[-1],(1,2,0)))

plt.subplot(1,3,3)
plt.axis("off")
plt.title("Fake images - non-saturating G loss")
plt.imshow(np.transpose(img_list_nonsat[-1],(1,2,0)))

plt.savefig('dcgan_comparison.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()

print("Saved: dcgan_comparison.png")