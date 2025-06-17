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
import math

DATA_DIR = "data/300"
NUM_EPOCHS = 5000
BATCH_SIZE = 64
NOISE_DIM = 120
NUM_WORKERS = 1
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
FEATURE_MATCHING_LAMBDA = 1
IMG_HEIGHT = 224
IMG_WIDTH = 296
LOG_STEPS = 1
CLIP_LAMBDA = 2
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.leaky = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.leaky(x + self.block(x))


class EnhancedGenerator(nn.Module):
    def __init__(self, noise_dim=120):
        super().__init__()
        self.init_h, self.init_w = 7, 9
        self.fc = nn.Linear(noise_dim, 512 * self.init_h * self.init_w)

        def up_block(in_channels, out_channels, use_attention=False):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                ResidualBlock(out_channels)
            ]
            if use_attention:
                layers.append(SelfAttention(out_channels))
            return nn.Sequential(*layers)

        self.main = nn.Sequential(
            nn.Unflatten(1, (512, self.init_h, self.init_w)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),

            up_block(512, 256, use_attention=True),   # Attention at 14x18
            up_block(256, 128),                       
            up_block(128, 64, use_attention=True),    # Attention at 56x74
            up_block(64, 32),
            up_block(32, 16),

            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )

        self.final_resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))

    def forward(self, x):
        x = self.main(self.fc(x))
        if x.shape[-2:] != (IMG_HEIGHT, IMG_WIDTH):
            x = self.final_resize(x)
        return x

class EnhancedDiscriminator(nn.Module):
    def __init__(self, input_shape=(3, IMG_HEIGHT, IMG_WIDTH)):
        super().__init__()

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(input_shape[0], 32, 4, stride=1, padding=1)),  # 112x148
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),  # 56x74
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 28x37
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            SelfAttention(128),  # 👁️ Attention at 28x37

            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),  # 14x18
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            SelfAttention(256),  # 👁️ Attention at 14x18
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
        patch_output = x  # preserve spatial score map (PatchGAN-style)
        global_out = self.fc(x.flatten(1))

        

        return global_out, patch_output, features
    
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

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
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

    train_sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    cpu_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                            num_workers=0, pin_memory=False, drop_last=True)

    generator = EnhancedGenerator(NOISE_DIM).to(device)
    discriminator = EnhancedDiscriminator((3, IMG_HEIGHT, IMG_WIDTH)).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    seed = torch.randn(16, NOISE_DIM, device=device)

    def preprocess_for_clip_batch(images):
        images = (images + 1) / 2
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        return (images - mean) / std

    ema_decay = 0.999
    r1_gamma = 2.0

    for epoch in range(NUM_EPOCHS):
        ema_generator = copy.deepcopy(generator).eval()
        
        para_loader = pl.ParallelLoader(cpu_loader, [device])
        loader = para_loader.per_device_loader(device)
        train_sampler.set_epoch(epoch)

        for step, batch_data in enumerate(loader):
            real_images = batch_data[0].to(device)
            batch_size = real_images.size(0)

            #### ========== DISCRIMINATOR STEP ==========
            d_optimizer.zero_grad()

            real_images.requires_grad_()
            real_logits, real_patch, real_feats = discriminator(real_images)

            d_loss_real = bce_loss(real_logits, torch.ones_like(real_logits))

            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_logits, fake_patch, _ = discriminator(fake_images.detach())
            d_loss_fake = bce_loss(fake_logits, torch.zeros_like(fake_logits))

            grads = torch.autograd.grad(outputs=real_logits, inputs=real_images,
                                        grad_outputs=torch.ones_like(real_logits),
                                        create_graph=True, retain_graph=True)[0]
            r1_penalty = grads.pow(2).sum([1, 2, 3]).mean()

            d_loss_patch = F.mse_loss(fake_patch.mean([2, 3]), real_patch.mean([2, 3]))
            d_loss = d_loss_real + d_loss_fake + r1_gamma * r1_penalty + 0.1 * d_loss_patch
            d_loss.backward()
            xm.optimizer_step(d_optimizer, barrier=True)

            #### ========== GENERATOR STEP ==========
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images = generator(noise)

            fake_logits, _, fake_feats = discriminator(fake_images)
            g_adv_loss = bce_loss(fake_logits, torch.ones_like(fake_logits))

            with torch.no_grad():
                real_feats = discriminator(real_images)[-1]
            fm_loss = sum(l1_loss(f.mean(0), r.mean(0)) for f, r in zip(fake_feats, real_feats))

            real_clip = clip_model.encode_image(preprocess_for_clip_batch(real_images)).detach()
            real_clip = real_clip / real_clip.norm(dim=-1, keepdim=True)
            fake_clip = clip_model.encode_image(preprocess_for_clip_batch(fake_images))
            fake_clip = fake_clip / fake_clip.norm(dim=-1, keepdim=True)
            clip_loss = 1 - F.cosine_similarity(fake_clip, real_clip, dim=-1).mean()

            g_loss = g_adv_loss + FEATURE_MATCHING_LAMBDA * fm_loss + CLIP_LAMBDA * clip_loss
            g_loss.backward()
            xm.optimizer_step(g_optimizer, barrier=True)

            update_ema(ema_generator, generator, ema_decay)

            #### ========== LOGGING ==========
            if step % LOG_STEPS == 0:
                print(
                    f"[Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}] "
                    f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}, "
                    f"CLIP_Loss: {clip_loss.item():.4f}, Time: {time.asctime()}"
                )

        xm.rendezvous('epoch_end')
        if xm.is_master_ordinal():
            generate_and_save_images(ema_generator, epoch + 1, seed, device)
            xm.save(generator.state_dict(), f"relaxed_generator1.pt")
            xm.save(discriminator.state_dict(), f"relaxed_discriminator1.pt")
            xm.save(ema_generator.state_dict(), f"ema_relaxed_generator1.pt")


def main():
    print(f"Starting resumed training from pre-trained checkpoints with relaxed discriminator and on-the-fly CLIP embeddings.")
    xmp.spawn(_mp_fn, args=(DATA_DIR,), start_method='spawn')


if __name__ == "__main__":
    main()