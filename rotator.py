import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageDraw


def rotate_resize_pad(image_path, output_dir, angles=[90, 180, 270], target_size=(300, 255), fill_color=(0, 0, 0)):
    os.makedirs(output_dir, exist_ok=True)

    original = Image.open(image_path).convert("RGB")
    basename = os.path.splitext(os.path.basename(image_path))[0]

    def resize_to_fit(image, max_size):
        return ImageOps.contain(image, max_size, method=Image.BICUBIC)

    def pad_to_exact(image, target_size, fill_color):
        w, h = image.size
        target_w, target_h = target_size
        delta_w = target_w - w
        delta_h = target_h - h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2)
        )
        return ImageOps.expand(image, padding, fill=fill_color)

    for angle in [0] + angles:
        rotated = original.rotate(angle, expand=True)
        resized = resize_to_fit(rotated, target_size)
        final_img = pad_to_exact(resized, target_size, fill_color)

        filename = os.path.join(output_dir, f"{basename}_rot{angle:03d}.png")
        final_img.save(filename)


def preprocess_image_torch(image_path, angles=[90, 180, 270], target_size=(255, 300), 
                          bottom_mask_height=5, left_mask_width=2):
    """
    Preprocess image with PyTorch tensors instead of TensorFlow
    
    Args:
        image_path: Path to the image file
        angles: List of rotation angles to apply
        target_size: Target size as (height, width)
        bottom_mask_height: Height of bottom mask in pixels
        left_mask_width: Width of left mask in pixels
    
    Returns:
        torch.Tensor: Stack of processed images with shape (N, H, W, C)
    """
    image = Image.open(image_path).convert("RGB")

    # Apply masks
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Bottom mask
    draw.rectangle([(0, height - bottom_mask_height), (width, height)], fill=(125, 125, 125))
    
    # Left mask
    draw.rectangle([(0, 0), (left_mask_width, height)], fill=(125, 125, 125))

    all_versions = []

    for angle in [0] + angles:
        # Rotate image
        rotated = image.rotate(angle, expand=True)

        # Resize to fit within target dimensions while maintaining aspect ratio
        rotated = ImageOps.contain(rotated, (target_size[1], target_size[0]))

        w, h = rotated.size
        pad_w = target_size[1] - w
        pad_h = target_size[0] - h
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )
        padded = ImageOps.expand(rotated, padding, fill=(125, 125, 125))

        
        image_array = np.array(padded)
        image_tensor = torch.from_numpy(image_array).float()
        
        # Normalize from [0, 255] to [-1, 1]
        image_tensor = (image_tensor / 127.5) - 1.0

        all_versions.append(image_tensor)

    # Stack all versions
    return torch.stack(all_versions)


def preprocess_directory_torch(
    image_dir,
    target_size=(255, 300),
    angles=[90, 180, 270],
    allowed_exts=[".png", ".jpg", ".jpeg"]
):
    """
    Preprocess all images in a directory using PyTorch
    
    Args:
        image_dir: Directory containing images
        target_size: Target size as (height, width)
        angles: List of rotation angles to apply
        allowed_exts: List of allowed file extensions
    
    Returns:
        tuple: (torch.Tensor of all images, list of filenames)
    """
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if os.path.splitext(fname.lower())[1] in allowed_exts
    ]

    all_images = []
    all_filenames = []

    for path in image_paths:
        batch_tensor = preprocess_image_torch(path, angles=angles, target_size=target_size)
        all_images.append(batch_tensor)

        base = os.path.splitext(os.path.basename(path))[0]
        all_filenames.extend([f"{base}_rot{angle:03d}" for angle in [0] + angles])

    result_tensor = torch.cat(all_images, dim=0)
    return result_tensor, all_filenames


def image_generator(image_paths, angles, target_size):
    """
    Generator function that yields processed images one by one
    
    Args:
        image_paths: List of image file paths
        angles: List of rotation angles to apply
        target_size: Target size as (height, width)
    
    Yields:
        numpy.ndarray: Individual processed images as numpy arrays
    """
    for path in image_paths:
        # Get tensor stack for this image
        image_stack = preprocess_image_torch(path, angles=angles, target_size=target_size)
        
        # Yield each image in the stack as a numpy array
        for i in range(image_stack.shape[0]):
            # Convert back to numpy array for compatibility
            yield image_stack[i].numpy()


# Additional utility functions for PyTorch compatibility

def torch_resize_and_pad(image_tensor, target_size, fill_value=125/127.5 - 1):
    """
    Resize and pad a PyTorch tensor image to target size
    
    Args:
        image_tensor: Input tensor of shape (H, W, C)
        target_size: Target size as (height, width)
        fill_value: Value to use for padding (normalized)
    
    Returns:
        torch.Tensor: Resized and padded image tensor
    """
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
   
    denorm_tensor = ((image_tensor + 1) * 127.5).clamp(0, 255).byte()
    
    if denorm_tensor.dim() == 3:
        image_pil = TF.to_pil_image(denorm_tensor)
    else:
        image_pil = TF.to_pil_image(denorm_tensor.unsqueeze(0))
    
    resized = ImageOps.contain(image_pil, (target_size[1], target_size[0]))
    
    w, h = resized.size
    pad_w = target_size[1] - w
    pad_h = target_size[0] - h
    padding = (
        pad_w // 2,
        pad_h // 2,
        pad_w - pad_w // 2,
        pad_h - pad_h // 2
    )
    padded = ImageOps.expand(resized, padding, fill=(125, 125, 125))
    
    result_tensor = torch.from_numpy(np.array(padded)).float()
    result_tensor = (result_tensor / 127.5) - 1.0
    
    return result_tensor


def batch_preprocess_torch(image_paths, angles, target_size, batch_size=32):
    """
    Process images in batches for memory efficiency
    
    Args:
        image_paths: List of image file paths
        angles: List of rotation angles to apply
        target_size: Target size as (height, width)
        batch_size: Number of images to process at once
    
    Yields:
        torch.Tensor: Batches of processed images
    """
    batch_images = []
    
    for path in image_paths:
        image_stack = preprocess_image_torch(path, angles=angles, target_size=target_size)
        
        for i in range(image_stack.shape[0]):
            batch_images.append(image_stack[i])
            
            if len(batch_images) == batch_size:
                yield torch.stack(batch_images)
                batch_images = []
    
    # Yield remaining images
    if batch_images:
        yield torch.stack(batch_images)