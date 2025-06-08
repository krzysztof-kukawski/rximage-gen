import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import layers, models

from rotator import image_generator

BATCH_SIZE = 4
images_120 = "data/300"
image_index = pd.read_csv(r"data/table.csv")
print(tf.test.is_built_with_cuda())
allowed_exts = [".png", ".jpg", ".jpeg"]
image_paths = [
    os.path.join(images_120, fname)
    for fname in os.listdir(images_120)[:500]
    if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
]
print("LEN!!!", len(image_paths))

dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(image_paths, angles=[], target_size=(225, 300)),
    output_signature=tf.TensorSpec(shape=(225, 300, 3), dtype=tf.float32),
)
batched_dataset = dataset.batch(BATCH_SIZE)
for batch in batched_dataset.take(1):
    print(batch.shape)


def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(29 * 38 * 512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((29, 38, 512)))

    model.add(
        layers.Conv2DTranspose(
            300, kernel_size=3, strides=2, padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            180, kernel_size=3, strides=2, padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            128, kernel_size=3, strides=2, padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            3,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )
    model.add(layers.Cropping2D(((4, 3), (2, 2))))
    return model


def make_discriminator_model():
    model = models.Sequential(
        [   layers.SpectralNormalization(
                layers.Conv2D(
                    32, 3, strides=2, padding="same", input_shape=[225, 300, 3]
                )
            ),
            layers.SpectralNormalization(
                layers.Conv2D(
                    64, 3, strides=2, padding="same", input_shape=[225, 300, 3]
                )
            ),
            layers.LeakyReLU(),
            layers.Dropout(0.2),
            layers.SpectralNormalization(
                layers.Conv2D(128, 3, strides=2, padding="same")
            ),
            layers.LeakyReLU(),
            layers.Dropout(0.2),
            layers.SpectralNormalization(
                layers.Conv2D(256, 3, strides=2, padding="same")
            ),
            layers.LeakyReLU(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(1),
        ]
    )
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output, gp, gp_weight=10.0):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp_weight * gp


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]

    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = tape.gradient(pred, interpolated)

    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)

    penalty = tf.reduce_mean((grads_norm - 1.0) ** 2)
    return penalty


generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=5e-4, beta_1=0.0, beta_2=0.9
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=5e-4, beta_1=0.0, beta_2=0.9
)

EPOCHS = 5000
noise_dim = 100
num_examples_to_generate = BATCH_SIZE


@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]

    for _ in range(5):
        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)

            gp = gradient_penalty(discriminator, real_images, fake_images)
            disc_loss = discriminator_loss(real_output, fake_output, gp)

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))  # type: ignore

    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))  # type: ignore

    return gen_loss, disc_loss


def generate_and_save_images_separately(
    model, epoch, test_input, output_dir="generated_images"
):
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0

    for i in range(predictions.shape[0]):
        image = predictions[i].numpy()

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        filename = os.path.join(epoch_dir, f"generated_{i:03d}.png")
        plt.imsave(filename, image)


def train(dataset, epochs, batch_size):
    total_images = 0
    for _ in dataset:
        total_images += 1
    total_images *= batch_size

    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch + 1}/{epochs}")
        start_epoch_time = time.time()

        processed_images = 0
        for batch_idx, image_batch in enumerate(dataset):
            batch_start_time = time.time()

            g_loss, d_loss = train_step(image_batch)
            processed_images += image_batch.shape[0]

            batch_time = time.time() - batch_start_time
            images_left = total_images - processed_images

            print(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}, "
                f"G loss: {g_loss:.4f}, D loss: {d_loss:.4f}, "
                f"Batch time: {batch_time:.2f}s, "
                f"Processed: {processed_images}/{total_images}, "
                f"Remaining: {images_left}"
            )

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

        if (epoch + 1) % 1 == 0:
            seed = tf.random.normal([num_examples_to_generate, noise_dim])
            generate_and_save_images_separately(generator, epoch + 1, seed)
            generator.save("gen.keras")
            discriminator.save("disc.keras")


def generate_and_show_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1.0) / 2.0 

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis("off")
    plt.show()


train(batched_dataset, EPOCHS, BATCH_SIZE)
generate_and_show_images(generator, EPOCHS, seed)
