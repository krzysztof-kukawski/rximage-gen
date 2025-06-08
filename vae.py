import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from rotator import image_generator

BATCH_SIZE = 4
images_120 = "data/600"
allowed_exts = [".png", ".jpg", ".jpeg"]
image_paths = [
    os.path.join(images_120, fname)
    for fname in os.listdir(images_120)[:6000]
    if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
]
NUM_SAMPLES = len(image_paths)
latent_dim = 64



def create_dataset(image_dir, batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(image_paths, angles=[0], target_size=(450, 600, 3)),
        output_signature=tf.TensorSpec(shape=(450, 600, 3), dtype=tf.float32),
    )
    return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()


def build_encoder(input_shape=(450, 600, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return tf.keras.Model(inputs, [z_mean, z_log_var], name="encoder")


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_decoder(latent_dim):
    decoder = tf.keras.models.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(57 * 75 * 128, activation="relu"),  
            layers.Reshape((57, 75, 128)),  
            layers.Conv2DTranspose(
                64, 3, strides=2, padding="same", activation="relu"
            ),  
            layers.Conv2DTranspose(
                32, 3, strides=2, padding="same", activation="relu"
            ), 
            layers.Conv2DTranspose(
                16, 3, strides=2, padding="same", activation="relu"
            ), 
            layers.Conv2DTranspose(
                3, 3, strides=1, padding="same", activation="sigmoid"
            ), 
            layers.Cropping2D(((3, 3), (0, 0))),  
        ]
    )

    return decoder


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampling()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampler([z_mean, z_log_var])
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.keras.losses.binary_crossentropy(
                data, reconstruction
            )

            reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[1, 2])
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


def save_reconstructed_images(model, dataset, epoch, output_dir="vae_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    for images in dataset.take(1):
        reconstructed = model(images, training=False)
        for i in range(min(5, images.shape[0])):
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(images[i].numpy())
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(reconstructed[i].numpy())
            axes[1].set_title("Reconstructed")
            axes[1].axis("off")
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_img_{i}.png"))
            plt.close()


def train_vae(image_dir, epochs=50, batch_size=16, save_interval=5):
    dataset = create_dataset(image_dir, batch_size=batch_size)

    encoder = build_encoder()
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        vae.fit(dataset, epochs=1, steps_per_epoch=NUM_SAMPLES // BATCH_SIZE)

        if epoch % save_interval == 0 or epoch == epochs:
            save_reconstructed_images(vae, dataset, epoch)
            vae.save("rx-image-gen.keras")


train_vae(image_dir="vae_img", epochs=900, batch_size=BATCH_SIZE)
