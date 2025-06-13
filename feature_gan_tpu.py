import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch_xla.core.xla_model as xm
from torch_xla.distributed.parallel_loader import MpDeviceLoader

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        self.fc = nn.Linear(noise_dim, 9 * 10 * 256, bias=False)
        self.bn0 = nn.BatchNorm1d(9 * 10 * 256)
        
        self.conv1_transpose = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1_transpose = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2_transpose = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2_transpose = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3_transpose = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn3_transpose = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4_transpose = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4_transpose = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.conv5_transpose = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5_transpose = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        
        self.final_conv = nn.Conv2d(16, 3, kernel_size=(64, 21), stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = x.view(-1, 256, 9, 10)
        
        # Block 1
        x = self.conv1_transpose(x)
        x = self.bn1_transpose(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Block 2
        x = self.conv2_transpose(x)
        x = self.bn2_transpose(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Block 3
        x = self.conv3_transpose(x)
        x = self.bn3_transpose(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4_transpose(x)
        x = self.bn4_transpose(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5_transpose(x)
        x = self.bn5_transpose(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        # Final layer
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, 3, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 64, 3, stride=2, padding=1))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
       
        self.fc = nn.Linear(128 * 29 * 38, 1)
        
    def forward(self, x, return_features=False):
        features = []
        
        x = self.conv1(x)
        x = self.leaky_relu(x)
        if return_features:
            features.append(x.clone())
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        if return_features:
            features.append(x.clone())
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if return_features:
            return x, features
        return x


def generator_loss_with_feature_matching(fake_images, real_images, discriminator, epoch):
    fake_output = discriminator(fake_images)
    adv_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
    
    with torch.no_grad():
        _, real_features = discriminator(real_images, return_features=True)
    _, fake_features = discriminator(fake_images, return_features=True)
    
    fm_losses = []
    for real_feat, fake_feat in zip(real_features, fake_features):
        real_mean = torch.mean(real_feat, dim=0)
        fake_mean = torch.mean(fake_feat, dim=0)
        fm_loss = torch.mean((real_mean - fake_mean) ** 2)
        fm_losses.append(fm_loss)
    
    fm_loss = sum(fm_losses)
    
    lambda_fm = min(1.0, epoch / 100.0)
    total_loss = adv_loss + lambda_fm * fm_loss
    
    return total_loss

# Device
device = xm.xla_device()

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 225, 300
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 64
EPOCHS = 2
SAVE_INTERVAL = 1
IMAGE_DIR = "gan_generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

images_120 = "data/300"
allowed_exts = [".png", ".jpg", ".jpeg"]
image_paths = [
    os.path.join(images_120, fname)
    for fname in os.listdir(images_120)[:2]
    if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
]

image_paths = image_paths 

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return transform(image)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', n_power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data), dim=0)
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data), dim=0)

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def train_step(real_images, generator, discriminator, g_optimizer, d_optimizer, criterion, epoch):
    batch_size = real_images.size(0)
    d_optimizer.zero_grad()
    real_output = discriminator(real_images)
    real_labels = torch.ones_like(real_output) * 0.9
    d_loss_real = criterion(real_output, real_labels)

    noise = torch.randn(batch_size, NOISE_DIM, device=device)
    fake_images = generator(noise)
    fake_output = discriminator(fake_images.detach())
    fake_labels = torch.zeros_like(fake_output)
    d_loss_fake = criterion(fake_output, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    xm.optimizer_step(d_optimizer)
    xm.mark_step()

    g_optimizer.zero_grad()
    noise = torch.randn(batch_size, NOISE_DIM, device=device)
    fake_images = generator(noise)
    g_loss = generator_loss_with_feature_matching(fake_images, real_images, discriminator, epoch)
    g_loss.backward()
    xm.optimizer_step(g_optimizer)
    xm.mark_step()

    return g_loss.item(), d_loss.item()

def generate_and_save_images(generator, epoch, seed):
    generator.eval()
    with torch.no_grad():
        predictions = generator(seed)
        predictions = (predictions + 1.0) / 2.0

        for i, img in enumerate(predictions[:24]):
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(IMAGE_DIR, f"epoch_{epoch:04d}_img_{i:02d}.png"))
    generator.train()

if __name__ == "__main__":
    dataset = CustomDataset(image_paths)
    cpu_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)
    dataloader = MpDeviceLoader(cpu_loader, device)

    generator = Generator(NOISE_DIM).to(device)
    discriminator = Discriminator().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0006, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0006, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    seed = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
    s = 1

    for epoch in range(EPOCHS):
        start_time = time.time()

        for step, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            g_loss, d_loss = train_step(real_images, generator, discriminator, g_optimizer, d_optimizer, criterion, epoch)
            print(f"Step {step+1}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}")

        print(f"Epoch {epoch+1}, Time: {time.time()-start_time:.2f}s")
        print(xm.get_memory_info(device))

        if (epoch + 1) % SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            torch.save(generator.state_dict(), f"feature_gan_gen{s}.pth")
            torch.save(discriminator.state_dict(), f"feature_gan_disc{s}.pth")
            s = 1 if s == 2 else 2
