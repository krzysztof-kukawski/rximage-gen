import os

import torch.nn.functional as F
import open_clip
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['XLA_USE_BF16'] = '1'
from torch.nn.utils import spectral_norm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import copy
import numpy as np
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, distributed
import torch.optim as optim
import torch.nn as nn
import torch
import time


DATA_DIR = "data/300"
NUM_EPOCHS = 5000
BATCH_SIZE = 128
NOISE_DIM = 120
NUM_WORKERS = 1
LEARNING_RATE = 0.00009
BETA1 = 0.5
BETA2 = 0.999
FEATURE_MATCHING_LAMBDA = 1
IMG_HEIGHT = 224
IMG_WIDTH = 296
LOG_STEPS = 1
CLIP_LAMBDA = 2

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.leaky = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        return self.leaky(x + self.block(x))


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.init_h, self.init_w = 7, 9
        self.fc = nn.Linear(noise_dim, 512 * self.init_h * self.init_w)

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                ResidualBlock(out_channels)
            )

        self.main = nn.Sequential(
            nn.Unflatten(1, (512, self.init_h, self.init_w)),
            nn.BatchNorm2d(512), nn.LeakyReLU(
                inplace=True, negative_slope=0.2),

            up_block(512, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 16),

            nn.Conv2d(16, 3, kernel_size=9, stride=1, padding=0),
            nn.Tanh()
        )

        self.final_resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))

    def forward(self, x):
        x = self.main(self.fc(x))
        if x.shape[-2:] != (IMG_HEIGHT, IMG_WIDTH):
            x = self.final_resize(x)
        return x


class RelaxedDiscriminator(nn.Module):
    def __init__(self, input_shape=(3, IMG_HEIGHT, IMG_WIDTH)):
        super(RelaxedDiscriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            spectral_norm(
                nn.Conv2d(input_shape[0], 32, 4, stride=1, padding=1)),  # 112x148
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),  # 56x74
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 28x37
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            spectral_norm(
                nn.Conv2d(128, 256, 4, stride=2, padding=1)),  # 14x18
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
        return x, features[-1:]


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
        self.image_paths = get_image_paths(data_dir)
        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return (self.transform(image),)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


def generate_and_save_images(generator, epoch, seed, device):
    generator.eval()
    with torch.no_grad():
        predictions = generator(seed).cpu() * 0.5 + 0.5
        os.makedirs("gan_generated_images", exist_ok=True)
        utils.save_image(
            predictions, f"gan_generated_images/epoch_{epoch:04d}.png", nrow=4)
    generator.train()


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def _mp_fn(rank, data_dir):
    device = xm.xla_device()
    world_size = xr.world_size()

    # Load CLIP model for encoding both real and fake images
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device).eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(data_dir, transform)
    if len(dataset) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return

    train_sampler = distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    cpu_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                            num_workers=0, pin_memory=False, drop_last=True)

    generator = Generator(NOISE_DIM).to(device)
    discriminator = RelaxedDiscriminator((3, IMG_HEIGHT, IMG_WIDTH)).to(device)

    generator.load_state_dict(torch.load("relaxed_generator.pt", map_location="cpu"))
    discriminator.load_state_dict(torch.load("relaxed_discriminator.pt", map_location="cpu"))

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    seed = torch.randn(16, NOISE_DIM, device=device)

    # Optimized CLIP preprocessing function (stays on device)
    def preprocess_for_clip_batch(images, device):
        """Optimized batch preprocessing for CLIP that stays on device"""
        images = (images + 1) / 2  # Scale to [0,1]
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images

    # Helper function to determine when to compute CLIP loss
    def should_compute_clip_loss(step, epoch):
        if epoch < 10:
            return step % 4 == 0  # Every 4 steps early on
        elif epoch < 50:
            return step % 2 == 0  # Every 2 steps mid-training
        else:
            return True  # Every step later

    for epoch in range(NUM_EPOCHS):
        para_loader = pl.ParallelLoader(cpu_loader, [device])
        train_sampler.set_epoch(epoch)
        loader = para_loader.per_device_loader(device)
        ema_generator = copy.deepcopy(generator)
        ema_generator.eval()
        ema_decay = 0.999
        r1_gamma = 0.1

        for step, batch_data in enumerate(loader):
            real_images = batch_data[0].to(device)
            current_batch_size = real_images.size(0)
            
            # === DISCRIMINATOR TRAINING ===
            d_optimizer.zero_grad()

            real_images.requires_grad_()
            real_output, _ = discriminator(real_images)
            d_loss_real = bce_loss(real_output, torch.ones_like(real_output))

            noise = torch.randn(current_batch_size, NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_output, _ = discriminator(fake_images.detach())
            d_loss_fake = bce_loss(fake_output, torch.zeros_like(fake_output))

            grads = torch.autograd.grad(outputs=real_output, inputs=real_images,
                                        grad_outputs=torch.ones_like(real_output),
                                        create_graph=True, retain_graph=True)[0]
            r1_penalty = grads.pow(2).sum([1, 2, 3]).mean()
            d_loss = d_loss_real + d_loss_fake + r1_gamma * r1_penalty
            d_loss.backward()
            xm.optimizer_step(d_optimizer, barrier=True)

            d_loss_scalar = xm.mesh_reduce(
                'd_loss_scalar', d_loss.detach(), lambda x: sum(x) / len(x))

            gen_inc = 1 if d_loss_scalar >= 0.5 else 3

            # === GENERATOR TRAINING ===
            for gen_step in range(gen_inc):
                g_optimizer.zero_grad()
                noise = torch.randn(current_batch_size, NOISE_DIM, device=device)
                fake_images = generator(noise)

                fake_output, fake_features = discriminator(fake_images)
                g_loss_adv = bce_loss(fake_output, torch.ones_like(fake_output))

                # Feature matching loss
                with torch.no_grad():
                    _, real_features = discriminator(real_images)
                g_loss_fm = sum(l1_loss(r.mean(0), f.mean(0)) for r, f in zip(real_features, fake_features))

                # CLIP loss (conditional) - now computed on-the-fly
                if should_compute_clip_loss(step, epoch):
                    with torch.no_grad():
                        # Encode real images to get CLIP embeddings
                        real_clip_images = preprocess_for_clip_batch(real_images, device)
                        real_clip_features = clip_model.encode_image(real_clip_images)
                        real_clip_features = real_clip_features / real_clip_features.norm(dim=-1, keepdim=True)

                    # Encode fake images (gradients enabled for generator training)
                    fake_clip_images = preprocess_for_clip_batch(fake_images, device)
                    fake_clip_features = clip_model.encode_image(fake_clip_images)
                    fake_clip_features = fake_clip_features / fake_clip_features.norm(dim=-1, keepdim=True)

                    # Compute cosine similarity loss
                    clip_loss = 1 - F.cosine_similarity(fake_clip_features, real_clip_features, dim=-1).mean()
                    clip_weight = CLIP_LAMBDA
                else:
                    clip_loss = torch.tensor(0.0, device=device)
                    clip_weight = 0.0

                g_loss = g_loss_adv + FEATURE_MATCHING_LAMBDA * g_loss_fm + clip_weight * clip_loss
                g_loss.backward()
                xm.optimizer_step(g_optimizer, barrier=True)
                
                # Update EMA less frequently
                if gen_step == 0:  # Only update on first generator step
                    update_ema(ema_generator, generator, ema_decay)

            if step % LOG_STEPS == 0:
                clip_loss_val = clip_loss.item() if isinstance(clip_loss, torch.Tensor) else 0.0
                print(
                    f"[Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}] "
                    f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}, "
                    f"CLIP_Loss: {clip_loss_val:.4f}, Time: {time.asctime()}")

        xm.rendezvous('epoch_end')
        if xm.is_master_ordinal():
            generate_and_save_images(ema_generator, epoch + 1, seed, device)
            xm.save(generator.state_dict(), f"relaxed_generator.pt")
            xm.save(discriminator.state_dict(), f"relaxed_discriminator.pt")
            xm.save(ema_generator.state_dict(), f"ema_relaxed_generator.pt")

def main():
    print(f"Starting resumed training from pre-trained checkpoints with relaxed discriminator and on-the-fly CLIP embeddings.")
    xmp.spawn(_mp_fn, args=(DATA_DIR,), start_method='spawn')


if __name__ == "__main__":
    main()