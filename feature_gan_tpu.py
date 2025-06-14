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
NUM_EPOCHS = 200
BATCH_SIZE = 200
NOISE_DIM = 100
NUM_WORKERS = os.cpu_count() 
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
FEATURE_MATCHING_LAMBDA = 10.0
IMG_HEIGHT = 224
IMG_WIDTH = 296

from torch.nn.utils import spectral_norm
def get_image_paths(data_dir):
    allowed_exts = [".png", ".jpg", ".jpeg"]
    return [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
    ]

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

            # Final layer to adjust size and channels
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(self.fc(x))


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

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Warning: Skipping corrupted image {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def get_image_paths():
    allowed_exts = [".png", ".jpg", ".jpeg"]
    return [
        os.path.join(DATA_DIR, fname)
        for fname in os.listdir(DATA_DIR)
        if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
    ]

def generate_and_save_images(generator, epoch, seed, device):
    generator.eval()
    with torch.no_grad():
        predictions = generator(seed.to(device))
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
    device = xm.xla_device()
    world_size = xr.world_size()

  
    image_paths = get_image_paths(data_dir)
    
    if xm.is_master_ordinal():
        print(f"Rank {rank} (master) found {len(image_paths)} images.")
    xm.rendezvous('image_paths_loaded')


    transform = transforms.Compose([
		transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
    
    dataset = CustomDataset(image_paths, transform)

    train_sampler = distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    
    cpu_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    
    para_loader = pl.ParallelLoader(cpu_loader, [device])
    train_loader = para_loader.per_device_loader(device)

    generator = Generator(NOISE_DIM).to(device)
    discriminator = Discriminator(input_shape=(3, IMG_HEIGHT, IMG_WIDTH)).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    
    

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=not xm.is_master_ordinal())
        
        d_loss_tracker = 0.0
        g_loss_tracker = 0.0

        for step, real_images in enumerate(pbar):
           
            d_loss.backward()
            xm.optimizer_step(d_optimizer, barrier=True)

           
            g_loss.backward()
            xm.optimizer_step(g_optimizer, barrier=True)
            
            d_loss_tracker += d_loss.detach()
            g_loss_tracker += g_loss.detach()

            if (step + 1) % LOG_STEPS == 0:
                avg_d_loss = xm.all_reduce(xm.REDUCE_SUM, d_loss_tracker) / (LOG_STEPS * world_size)
                avg_g_loss = xm.all_reduce(xm.REDUCE_SUM, g_loss_tracker) / (LOG_STEPS * world_size)
                
                if xm.is_master_ordinal():
                    pbar.set_postfix(D_Loss=avg_d_loss.item(), G_Loss=avg_g_loss.item(), Step=step)
                
                # Reset trackers
                d_loss_tracker = 0.0
                g_loss_tracker = 0.0

        # End of epoch actions
        xm.rendezvous('epoch_end')
        if xm.is_master_ordinal():
            generate_and_save_images(generator, epoch + 1, seed, device)
            # Save checkpoints
            xm.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pt")
            xm.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pt")



def main():
    
    print(f"Starting training process for data in: {DATA_DIR}")

    
    xmp.spawn(_mp_fn, args=(DATA_DIR,), start_method='spawn')


if __name__ == "__main__":
    main()