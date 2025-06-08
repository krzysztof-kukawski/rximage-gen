import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


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


def preprocess_image_tf(image_path, angles=[90, 180, 270], target_size=(255, 300)):
    # Load image using PIL
    image = Image.open(image_path).convert("RGB")

    all_versions = []

    for angle in [0] + angles:
        rotated = image.rotate(angle, expand=True)

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
        padded = ImageOps.expand(rotated, padding, fill=(0, 0, 0))

        image_tensor = (tf.convert_to_tensor(np.array(padded), dtype=tf.float32) / 255)

        all_versions.append(image_tensor)

    return tf.stack(all_versions)


def preprocess_directory_tf(
        image_dir,
        target_size=(255, 300),
        angles=[90, 180, 270],
        allowed_exts=[".png", ".jpg", ".jpeg"]
):
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if os.path.splitext(fname.lower())[1] in allowed_exts
    ]

    all_images = []
    all_filenames = []

    for path in image_paths:
        batch_tensor = preprocess_image_tf(path, angles=angles, target_size=target_size)
        all_images.append(batch_tensor)

        base = os.path.splitext(os.path.basename(path))[0]
        all_filenames.extend([f"{base}_rot{angle:03d}" for angle in [0] + angles])

    result_tensor = tf.concat(all_images, axis=0)
    return result_tensor, all_filenames


def image_generator(image_paths, angles, target_size):
    for path in image_paths:
        yield from preprocess_image_tf(path, angles=angles, target_size=target_size)
