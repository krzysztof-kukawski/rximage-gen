import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from rotator import image_generator

IMG_HEIGHT, IMG_WIDTH = 225, 300
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 42
EPOCHS = 1000
SAVE_INTERVAL = 3  
IMAGE_DIR = "gan_generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
images_120 = "data/300"
allowed_exts = [".png", ".jpg", ".jpeg"]
image_paths = [
    os.path.join(images_120, fname)
    for fname in os.listdir(images_120)
    if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
]


def load_dataset():
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(image_paths, angles=[], target_size=(225, 300)),
        output_signature=tf.TensorSpec(shape=(225, 300, 3), dtype=tf.float32),
    )
    batched_dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return batched_dataset


dataset = load_dataset()


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(29 * 38 * 100, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((29, 38, 100)))

    model.add(layers.Conv2DTranspose(50, 3, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, 3, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            3, 3, strides=2, padding="same", use_bias=False, activation="tanh"
        )
    )

    model.add(layers.Cropping2D(((4, 3), (2, 2))))
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
    model.add(layers.Conv2D(32, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(16, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(8, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(8, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(4, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(
        zip(grads_disc, discriminator.trainable_variables)
    )

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)
    predictions = (predictions + 1.0) / 2.0  

    for i, img in enumerate(predictions):
        path = os.path.join(IMAGE_DIR, f"epoch_{epoch:04d}_img_{i:02d}.png")
        plt.imsave(path, img.numpy())


def train(dataset, epochs):
    seed = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    for epoch in range(epochs):
        start = time.time()
        step = 0
        for image_batch in dataset:
            step += 1

            g_loss, d_loss = train_step(image_batch)
            print(
                f"Step {step}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Time: {time.time()-start:.2f}s"
            )

        print(
            f"Epoch {epoch+1}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Time: {time.time()-start:.2f}s"
        )

        if (epoch + 1) % SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            generator.save("simple_gan_gen.keras")
            discriminator.save("simple_gan_disc.keras")


train(dataset, EPOCHS)
