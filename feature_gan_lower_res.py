import os
import time
from keras import Model
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from rotator import image_generator
os.chdir(os.path.dirname(__file__))
tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5500)])
    except RuntimeError as e:
        print(e)
IMG_HEIGHT, IMG_WIDTH = 225, 300
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 64
EPOCHS = 1000
SAVE_INTERVAL = 1
IMAGE_DIR = "gan_generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
images_120 = "data/120"
allowed_exts = [".png", ".jpg", ".jpeg"]
image_paths = [
    os.path.join(images_120, fname)
    for fname in os.listdir(images_120)
    if os.path.splitext(fname.lower())[1] in allowed_exts and "NLMIMAGE" in fname
]

class MinibatchStdDev(layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x):
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        variance = tf.reduce_mean((x - mean) ** 2, axis=0, keepdims=True)
        stddev = tf.sqrt(variance + self.epsilon)

        # Average over feature maps and spatial dims (H, W, C)
        mean_stddev = tf.reduce_mean(stddev)

        # Create a feature map filled with this value
        shape = tf.shape(x)
        stddev_feature = tf.fill([shape[0], shape[1], shape[2], 1], mean_stddev)

        # Concatenate with input
        return tf.concat([x, stddev_feature], axis=-1)


def load_dataset():
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(image_paths, angles=[90, 180, 270], target_size=(90, 120)),
        output_signature=tf.TensorSpec(shape=(90, 120, 3), dtype=tf.float16),
    )
    batched_dataset = dataset.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return batched_dataset


dataset = load_dataset()


def make_generator_model():
    model = tf.keras.Sequential(name="Hybrid_Generator")

    # Project and reshape latent vector to 6×8×256
    model.add(layers.Input((NOISE_DIM,)))
    model.add(layers.Dense(6 * 8 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((6, 8, 256)))  # → (6, 8, 256)

    # Upsample 1 → (12, 16, 128)
    model.add(layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Upsample 2 → (24, 32, 64)
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Upsample 3 → (48, 64, 32)
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Upsample 4 → (96, 128, 16)
    model.add(layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Final Conv → Downsample to (90, 120, 3)
    # (96 − 7 + 1 = 90), (128 − 9 + 1 = 120)
    model.add(layers.Conv2D(3, kernel_size=(7, 9), strides=1, padding='valid', activation='tanh'))

    return model
    


def make_discriminator_model():
    inputs = tf.keras.Input(shape=(90, 120, 3))
    x = layers.RandomRotation(factor=0.0417)(inputs) 
    
    feature_extractors = []
    x = layers.SpectralNormalization(layers.Conv2D(16, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.SpectralNormalization(layers.Conv2D(32, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    feature_extractors.append(Model(inputs, x))  # after third block
    
    x = layers.SpectralNormalization(layers.Conv2D(64, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.SpectralNormalization(layers.Conv2D(128, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    feature_extractors.append(Model(inputs, x))  # after third block

    # Conv Block 4
    x = layers.SpectralNormalization(layers.Conv2D(256, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

   
    feature_extractors.append(Model(inputs, x))  # after fifth block
    x = MinibatchStdDev()(x)
    # Global pooling and final dense output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1)(x)

    # Final full model
    model = Model(inputs, outputs)

    return model, feature_extractors

@tf.function
def generator_loss_with_feature_matching(fake_images, real_images, discriminator, feature_extractors, epoch):
    fake_output = discriminator(fake_images, training=True)
    adv_loss = bce_loss(tf.ones_like(fake_output), fake_output)

    fm_losses = []
    for extractor in feature_extractors:
        real_feats = extractor(real_images, training=False)
        fake_feats = extractor(fake_images, training=False)

        fm_loss = tf.reduce_mean(tf.square(
            tf.reduce_mean(real_feats, axis=0) - tf.reduce_mean(fake_feats, axis=0)
        ))
        fm_losses.append(fm_loss)

    fm_loss = tf.reduce_sum(tf.convert_to_tensor(fm_losses, dtype=tf.float32))
    
    λ = tf.minimum(1.0, epoch / 100.0)
    total_loss = adv_loss + λ * fm_loss
    return total_loss


discriminator, feature_extractors = make_discriminator_model()

generator = make_generator_model()
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0006, beta_1=0.5, beta_2=0.999
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0006, beta_1=0.5, beta_2=0.999
)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images, epoch):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    noise = noise + tf.random.normal(tf.shape(noise), mean=0.0, stddev=0.05)
    with tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        disc_loss_real = bce_loss(real_labels, real_output)
        disc_loss_fake = bce_loss(fake_labels, fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)
        gen_loss = generator_loss_with_feature_matching(
            fake_images, real_images, discriminator, feature_extractors, epoch
        )


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)
    predictions = (predictions + 1.0) / 2.0  

    for i, img in enumerate(predictions[:24]):
        path = os.path.join(IMAGE_DIR, f"epoch_{epoch:04d}_img_{i:02d}.png")
        plt.imsave(path, img.numpy())


def train(dataset, epochs):
    seed = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    s = 1
    for epoch in range(epochs):
        start = time.time()
        step = 0
        for image_batch in dataset:
            batch_start = time.time()
            step += 1

            g_loss, d_loss = train_step(image_batch, epoch)
            print(
                f"Step {step}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Batch time: {time.time()-batch_start:.2f}s"
            )

        print(
            f"Epoch {epoch+1}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Time: {time.time()-start:.2f}s"
        )
        if (epoch + 1) % SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            generator.save(f"feature_gan_gen{s}.keras")
            discriminator.save(f"feature_gan_disc{s}.keras")
            s +=1
            if s == 3:
                s = 1

train(dataset, EPOCHS)
