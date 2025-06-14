import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

DATA_DIR = "data/300"
NUM_EPOCHS = 1200
BATCH_SIZE = 64  # Reduced from 200 to 64 (divisible by 8)
NOISE_DIM = 100
NUM_WORKERS = 1
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
FEATURE_MATCHING_LAMBDA = 10.0
IMG_HEIGHT = 224
IMG_WIDTH = 296
LOG_STEPS = 10  # Reduced logging frequency
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.init_h, self.init_w = 7, 9
        self.fc = nn.Linear(noise_dim, 512 * self.init_h * self.init_w)

        self.main = nn.Sequential(
            # Unflatten and prepare
            nn.Unflatten(1, (512, self.init_h, self.init_w)),
            nn.BatchNorm2d(512), nn.ReLU(True),

            # (512, 7, 9) -> (256, 14, 18)
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            
            # (256, 14, 18) -> (128, 28, 36)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            
            # (128, 28, 36) -> (64, 56, 72)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),

            # (64, 56, 72) -> (32, 112, 144)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),

            # (32, 112, 144) -> (16, 224, 288)
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(True),

            # Final layer - use padding to get exact size
            nn.Conv2d(16, 3, kernel_size=9, stride=1, padding=0),  # This should output (224, 280)
            nn.Tanh()
        )
        
        # Add a final resize layer if needed
        self.final_resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))

    def forward(self, x):
        x = self.main(self.fc(x))
        # Apply resize if the dimensions don't match exactly
        if x.shape[-2:] != (IMG_HEIGHT, IMG_WIDTH):
            x = self.final_resize(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, IMG_HEIGHT, IMG_WIDTH)):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(input_shape[0], 64, 4, stride=2, padding=1)), # 112x148
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)), # 56x74
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)), # 28x37
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)), # 14x18
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = int(np.prod(dummy_output.shape[1:]))

        self.fc = nn.Linear(flattened_size, 1)

    def forward(self, x):
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, features


def get_image_paths(data_dir):
    allowed_exts = [".png", ".jpg", ".jpeg"]
    return [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
    ]


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        # Load paths immediately
        self.image_paths = get_image_paths(data_dir)
        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return (self.transform(image),)  # Return as tuple
        except Exception as e:
            print(f"Warning: Skipping corrupted image {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def generate_and_save_images(generator, epoch, seed, device):
    generator.eval()
    with torch.no_grad():
        predictions = generator(seed)
        predictions = predictions.cpu()
        predictions = predictions * 0.5 + 0.5
        save_path = "gan_generated_images"
        os.makedirs(save_path, exist_ok=True)
        utils.save_image(
            predictions,
            os.path.join(save_path, f"epoch_{epoch:04d}.png"),
            nrow=4,
            normalize=False
        )
    generator.train()


def _mp_fn(rank, data_dir):
    # Setup process
    device = xm.xla_device()
    world_size = xr.world_size()

    # Data loading setup - FIXED
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CustomDataset(data_dir, transform)  # FIXED: pass data_dir, not image_paths
    
    if len(dataset) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return
    
    train_sampler = distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    cpu_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    # Models & Optimizers
    generator = Generator(NOISE_DIM).to(device)
    discriminator = Discriminator(input_shape=(3, IMG_HEIGHT, IMG_WIDTH)).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # FIXED: Create seed on the correct device
    seed = torch.randn(16, NOISE_DIM, device=device)

    for epoch in range(NUM_EPOCHS):
        para_loader = pl.ParallelLoader(cpu_loader, [device])
        
        train_sampler.set_epoch(epoch)
        train_loader_for_epoch = para_loader.per_device_loader(device)

        if xm.is_master_ordinal():
            pbar = tqdm(desc=f"Epoch {epoch+1}")
        
        step = 0
        for batch_data in train_loader_for_epoch:
            real_images = batch_data[0]  # Unpack here instead
            # === Train Discriminator ===
            d_optimizer.zero_grad()

            # 1. Loss on real images
            real_output, _ = discriminator(real_images)
            d_loss_real = bce_loss(real_output, torch.ones_like(real_output))

            # 2. Loss on fake images
            noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_output, _ = discriminator(fake_images.detach())
            d_loss_fake = bce_loss(fake_output, torch.zeros_like(fake_output))

            # 3. Combine losses
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            xm.optimizer_step(d_optimizer, barrier=True)

            # === Train Generator ===
            g_optimizer.zero_grad()
            
            # Re-run forward pass for generator
            noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
            fake_images_for_g = generator(noise)
            fake_output_for_g, fake_features = discriminator(fake_images_for_g)
            
            # Adversarial Loss
            g_loss_adv = bce_loss(fake_output_for_g, torch.ones_like(fake_output_for_g))

            # Feature Matching Loss
            with torch.no_grad():
                _, real_features = discriminator(real_images)
                
            g_loss_fm = 0
            for real_feat, fake_feat in zip(real_features, fake_features):
                g_loss_fm += l1_loss(real_feat.mean(0), fake_feat.mean(0))

            # Combine generator losses
            g_loss = g_loss_adv + FEATURE_MATCHING_LAMBDA * g_loss_fm
            g_loss.backward()
            xm.optimizer_step(g_optimizer, barrier=True)
            
            step += 1
            if xm.is_master_ordinal() and step % LOG_STEPS == 0:
                print(
                    f"[Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}] "
                    f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}, "
                    f"Time: {time.asctime()}"
                )
                pbar.set_postfix(D_Loss=d_loss.item(), G_Loss=g_loss.item())
                pbar.update(LOG_STEPS)
        
        # End of epoch actions
        xm.rendezvous('epoch_end')
        if xm.is_master_ordinal():
            generate_and_save_images(generator, epoch + 1, seed, device)
            xm.save(generator.state_dict(), f"generator.pt")
            xm.save(discriminator.state_dict(), f"discriminator.pt")


def main():
    print(f"Starting training process for data in: {DATA_DIR}")
    xmp.spawn(_mp_fn, args=(DATA_DIR,), start_method='spawn')


if __name__ == "__main__":
    main()