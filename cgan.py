import os
import random
import torch.nn.functional as F
import open_clip
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['XLA_USE_BF16'] = '1'
from torch.nn.utils import spectral_norm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, distributed
import torch.optim as optim
import torch.nn as nn
import torch
import time
import math
import json
import matplotlib.pyplot as plt
DATA_DIR = "data/300"
NUM_EPOCHS = 5000
BATCH_SIZE = 64
NOISE_DIM = 120
NUM_WORKERS = 8
LEARNING_RATE = 0.0002
BETA1 = 0.0
BETA2 = 0.999
FEATURE_MATCHING_LAMBDA = 1
IMG_HEIGHT = 224
IMG_WIDTH = 296
LOG_STEPS = 1
CLIP_LAMBDA = 5
METADATA_PATH = 'data/rximagesAll.json'

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
    def __init__(self, noise_dim=120, cond_dim=0):  # <- new param
        super().__init__()
        self.init_h, self.init_w = 7, 9
        self.total_input_dim = noise_dim + cond_dim  # <- combined input
        self.fc = nn.Linear(self.total_input_dim, 512 * self.init_h * self.init_w)

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

    def forward(self, z, c=None):
        if c is not None:
            # z should be [batch_size, noise_dim], c should be [batch_size, cond_dim]
            combined_input = torch.cat([z, c], dim=1)  # [batch_size, noise_dim + cond_dim]
        else:
            combined_input = z
            
        # Pass through FC layer
        x = self.fc(combined_input)  # This should work now
        x = self.main(x)
        
        if x.shape[-2:] != (IMG_HEIGHT, IMG_WIDTH):
            x = self.final_resize(x)
        return x


class EnhancedDiscriminator(nn.Module):
    def __init__(self, input_shape=(3, IMG_HEIGHT, IMG_WIDTH), cond_dim=512):
        super().__init__()
        self.cond_dim = cond_dim
        self.input_shape = input_shape

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(input_shape[0], 32, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.0),

            spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.0),

            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.0),
            SelfAttention(128),

            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.0),
            SelfAttention(256),
        )

        # FIXED: Calculate feature dimension more robustly
        self.feature_dim = None
        self.uncond_fc = None
        self.embed_proj = None
        
        # We'll initialize these in the first forward pass
        self._initialized = False

    def _initialize_layers(self, x):
        """Initialize layers dynamically based on actual input"""
        if self._initialized:
            return
            
        
        # Get actual feature dimension
        with torch.no_grad():
            dummy_out = self.conv_layers(x[:1])  # Use actual input shape
            self.feature_dim = int(np.prod(dummy_out.shape[1:]))

        # Initialize layers
        self.uncond_fc = nn.Linear(self.feature_dim, 1).to(x.device)
        self.embed_proj = nn.Linear(self.cond_dim, self.feature_dim).to(x.device)
        
        
        
        self._initialized = True

    def forward(self, x, c):
        # Initialize layers if needed
        self._initialize_layers(x)
        
        B = x.size(0)
        x = self.conv_layers(x)
        x_flat = x.view(B, -1)  # h(x)

        # Unconditional score
        uncond_out = self.uncond_fc(x_flat)

        # Conditional projection
        c_proj = self.embed_proj(c)  # V(c)
        cond_out = torch.sum(x_flat * c_proj, dim=1, keepdim=True)  # h(x)^T * V(c)

        # Final score
        final_out = uncond_out + cond_out

        return final_out, x, []
def get_image_paths(data_dir):
    allowed_exts = [".png", ".jpg", ".jpeg"]
    return [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
    ]

def get_description(pill):
    if pill:
        desc = ''.join([f'{i} {j}, ' for i, j in pill['mpc'].items()])
        if pill['name']:
            full_desc = pill['name'] + ' ' + desc
        elif pill['ingredients']['active']:
            full_desc = pill['ingredients']['active'][0] + ' ' + desc
        else:
            full_desc = pill['labeler'] + ' ' + desc
    else:
        full_desc = ''
    return full_desc

def load_metadata():
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

def build_description_lookup(metadata):
    return {pill['nlmImageFileName']: get_description(pill) for pill in metadata}

def build_aligned_descriptions(image_paths, description_lookup):
    return [
        description_lookup[os.path.basename(path)]
        for path in image_paths
        if os.path.basename(path) in description_lookup
    ]
class PillDataset(Dataset):
    def __init__(self, image_paths, clip_preprocess, text_embeddings, descriptions):
        self.image_paths = image_paths
        self.clip_preprocess = clip_preprocess
        self.text_embeddings = text_embeddings
        self.descriptions = descriptions

        self.tensor_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_tensor = self.tensor_transform(image)
        text_embedding = self.text_embeddings[idx]
        description = self.descriptions[idx]
        return image_tensor, text_embedding, description

def clip_similarity_loss(fake_images, descriptions, clip_model, tokenizer, device):
    clip_model.eval()  # Keep frozen

    # Step 1: Preprocess images
    image_inputs = preprocess_for_clip_batch(fake_images)  # [B, 3, 224, 224]

    # Step 2: Encode text
    tokens = tokenizer(descriptions).to(device)

    # Step 3: Compute embeddings
    image_embeds = clip_model.encode_image(image_inputs)
    text_embeds = clip_model.encode_text(tokens)

    # Step 4: Normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Step 5: Similarity loss
    similarity = (image_embeds * text_embeds).sum(dim=1)  # shape: [batch]
    loss = 1 - similarity.mean()  # maximize similarity → minimize 1 - sim

    return loss
def generate_and_save_known_text(generator, epoch, clip_model, tokenizer, descriptions, device, num_samples=16, postfix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap

    generator = generator.to(device).eval()
    clip_model = clip_model.to(device).eval()

    sampled = random.sample(descriptions, num_samples)

    tokens = tokenizer(sampled).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(tokens)

    noise = torch.randn((num_samples, NOISE_DIM), device=device)
    condition = text_embeddings
    inputs = torch.cat([noise, condition], dim=1)

    with torch.no_grad():
        outputs = generator(inputs)
        xm.mark_step()  # For TPU
        fake_images = outputs.cpu() * 0.5 + 0.5

    # Save grid
    nrow, ncol = 4, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    axes = axes.flatten()

    for i in range(num_samples):
        img = fake_images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        wrapped = textwrap.fill(sampled[i], width=30)
        axes[i].set_title(wrapped, fontsize=8)

    for j in range(num_samples, nrow * ncol):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs("cgan_generated_images", exist_ok=True)
    filename = f"cgan_generated_images/epoch_{epoch:04d}{postfix}_known_text.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[✔] Saved generated samples with known descriptions to {filename}")

    generator.train()

    metadata = {
        "epoch": epoch,
        "descriptions": sampled,
        "embeddings": text_embeddings.cpu().numpy().tolist()
    }
    with open(filename.replace('.png', '.json'), 'w') as f:
        json.dump(metadata, f, indent=2)



def weighted_generator_loss( d_outputs, criterion, temperature=1.0):
    """
    Generator loss weighted by how fake the discriminator thinks each sample is.

    Args:
        generator: your generator model
        discriminator: your discriminator model
        z: latent vectors (batch_size, latent_dim)
        criterion: usually nn.BCELoss()
        temperature: softmax temperature for weighting

    Returns:
        scalar loss value
    """
    

    # Compute per-sample generator loss (goal: fool the discriminator into thinking it's real)
    target_real = torch.ones_like(d_outputs)  # Generator wants D(fake) → 1
    individual_losses = criterion(d_outputs, target_real)  # shape: [batch_size]

    # Compute weights: lower D output → worse sample → higher weight
    # We'll use softmax over "fakeness" = (1 - D_output)
    fakeness_scores = 1.0 - d_outputs.detach()  # detach so D doesn't get gradients
    weights = F.softmax(fakeness_scores / temperature, dim=0)  # shape: [batch_size]

    # Weighted sum of losses
    weighted_loss = torch.sum(weights * individual_losses)

    return weighted_loss

def _mp_fn(rank, data_dir):
    device = xm.xla_device()
    world_size = xr.world_size()

    # Load CLIP (text encoder only)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Load metadata and build lookup
    metadata = load_metadata()
    desc_lookup = build_description_lookup(metadata)

    # Prepare data
    image_paths = get_image_paths(DATA_DIR)
    metadata = load_metadata()
    description_lookup = build_description_lookup(metadata)
    aligned_descriptions = build_aligned_descriptions(image_paths, description_lookup)

    if not image_paths:
        print(f"[Rank {rank}] ERROR: No data in {data_dir}")
        return

    # Tokenize and encode all text descriptions once
    tokenized = open_clip.tokenize(aligned_descriptions).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(tokenized).cpu()

    # Construct dataset and loader
    dataset = PillDataset(image_paths, preprocess, text_embeddings, aligned_descriptions)
    if len(dataset) == 0:
        print(f"ERROR: No data in {data_dir}")
        return

    train_sampler = distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
    )
    cpu_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                            num_workers=0, pin_memory=False, drop_last=True)

    cond_dim = 512  # CLIP ViT-B-32 text embedding dim
    generator = EnhancedGenerator(noise_dim=NOISE_DIM, cond_dim=cond_dim) 
    discriminator = EnhancedDiscriminator((3, IMG_HEIGHT, IMG_WIDTH), cond_dim)

    generator.to(device)
    discriminator.to(device)

    pretrained_dict = torch.load("checkpoints/relaxed_generator5.pt", map_location="cpu")
    model_dict = generator.state_dict()

    # 1. Filter out any new keys in the generator (e.g., condition_fc)
    checkpoint = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    old_fc_weight = checkpoint['fc.weight']  # shape [32256, 120]
    old_fc_bias = checkpoint['fc.bias']      # shape [32256]
    new_fc_weight = generator.state_dict()['fc.weight']  # shape [32256, 632]
    new_fc_bias = generator.state_dict()['fc.bias']      # shape [32256]

    # Copy old weights into first 120 columns
    new_fc_weight[:, :120] = old_fc_weight
    # Leave remaining columns as initialized (random)

    # Copy bias
    new_fc_bias[:] = old_fc_bias

    # Update checkpoint dict
    checkpoint['fc.weight'] = new_fc_weight
    checkpoint['fc.bias'] = new_fc_bias

    # Load full checkpoint now
    generator.load_state_dict(checkpoint)
    # 2. Update
    """model_dict.update(pretrained_dict)
    generator.load_state_dict(model_dict)"""
    
    pretrained_dict = torch.load("checkpoints/relaxed_discriminator5.pt", map_location="cpu")
    model_dict = discriminator.state_dict()

    # 1. Filter out any new keys in the generator (e.g., condition_fc)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. Update
    model_dict.update(pretrained_dict)
    discriminator.load_state_dict(model_dict)

    g_optim = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_optim = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()


    for epoch in range(NUM_EPOCHS):
        para_loader = pl.ParallelLoader(cpu_loader, [device])
        loader = para_loader.per_device_loader(device)
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(loader):
            real_images, cond_embed, raw_desc = batch
            real_images = real_images.to(device)

            # Tokenize + embed
            with torch.no_grad():
                cond_embed = cond_embed / cond_embed.norm(dim=-1, keepdim=True)
            
            ### ========== DISCRIMINATOR STEP ==========
            d_optim.zero_grad()

            real_images.requires_grad_()
            real_logits, real_patch, real_feats = discriminator(real_images, cond_embed)
            d_loss_real = bce_loss(real_logits, torch.ones_like(real_logits))

            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            
            fake_images = generator(noise, cond_embed)
            fake_logits, fake_patch, _ = discriminator(fake_images.detach(), cond_embed)
            d_loss_fake = bce_loss(fake_logits, torch.zeros_like(fake_logits))
            
            grads = torch.autograd.grad(
                outputs=real_logits,
                inputs=real_images,
                grad_outputs=torch.ones_like(real_logits),
                create_graph=True, retain_graph=True
            )[0]
            r1_penalty = grads.pow(2).sum([1, 2, 3]).mean()

            patch_loss = F.mse_loss(fake_patch.mean([2, 3]), real_patch.mean([2, 3]))

            d_loss = d_loss_real + d_loss_fake + 1 * r1_penalty + 0.1 * patch_loss
            d_loss.backward()
            xm.optimizer_step(d_optim, barrier=True)

            ### ========== GENERATOR STEP ==========
            g_optim.zero_grad()
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            noise += 0.02 * torch.randn_like(noise)  # noise smoothing
            fake_images = generator(noise, cond_embed)

            fake_logits, _, fake_feats = discriminator(fake_images, cond_embed)
            g_adv_loss = weighted_generator_loss(fake_logits, bce_loss, temperature=1)

            with torch.no_grad():
                _, _, real_feats = discriminator(real_images, cond_embed)
            fm_loss = sum(l1_loss(f.mean(0), r.mean(0)) for f, r in zip(fake_feats, real_feats))

            
            clip_loss = clip_similarity_loss(fake_images, raw_desc, clip_model, open_clip.tokenize, device)

            clip_weight = CLIP_LAMBDA * (1 + math.sin(epoch / 120 * math.pi)) / 2
            g_loss = g_adv_loss + FEATURE_MATCHING_LAMBDA * fm_loss + clip_weight * clip_loss

            g_loss.backward()
            xm.optimizer_step(g_optim, barrier=True)

            if step % LOG_STEPS == 0:
                print(
                    f"[Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}] "
                    f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}, "
                    f"CLIP_Loss: {clip_loss.item():.4f}, Time: {time.asctime()}"
                )

        xm.rendezvous('epoch_end')
        if xm.is_master_ordinal():
            generate_and_save_known_text(generator, epoch,clip_model, open_clip.tokenize, aligned_descriptions, device)

            xm.save(generator.state_dict(), f"checkpoints/cgan_generator.pt")
            xm.save(discriminator.state_dict(), f"checkpoints/cgan_discriminator.pt")


def preprocess_for_clip_batch(images):
    images = (images + 1) / 2
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def main():
    print("Starting CLIP-conditioned cGAN training")
    xmp.spawn(_mp_fn, args=(DATA_DIR,), start_method='spawn')


if __name__ == "__main__":
    main()