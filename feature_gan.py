import os
import time
from keras import Model
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from rotator import image_generator

IMG_HEIGHT, IMG_WIDTH = 225, 300
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 25
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
    batched_dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return batched_dataset


dataset = load_dataset()


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(29 * 38 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((29, 38, 256)))

    model.add(layers.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, 3, strides=2, padding="same", use_bias=False))
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
    inputs = tf.keras.Input(shape=(225, 300, 3))  # type: ignore
    x = layers.SpectralNormalization(layers.Conv2D(64, 3, strides=2, padding="same"))(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.SpectralNormalization(layers.Conv2D(80, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.SpectralNormalization(layers.Conv2D(100, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.SpectralNormalization(layers.Conv2D(128, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.SpectralNormalization(layers.Conv2D(180, 3, strides=2, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    feature_layer = x
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    feature_extractor = Model(inputs, feature_layer)

    return model, feature_extractor


def generator_loss_with_feature_matching(fake_images, real_images, discriminator, feature_extractor):
    fake_output = discriminator(fake_images, training=True)
    adv_loss = bce_loss(tf.ones_like(fake_output), fake_output)

    fake_features = feature_extractor(fake_images)
    real_features = feature_extractor(real_images)

    fm_loss = tf.reduce_mean(
        tf.square(
            tf.reduce_mean(real_features, axis=0) - tf.reduce_mean(fake_features, axis=0)
        )
    )

    λ = 10.0
    total_loss = adv_loss + λ * fm_loss
    return total_loss


discriminator, feature_extractor = make_discriminator_model()


generator = make_generator_model()
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.999
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.999
)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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
            fake_images, real_images, discriminator, feature_extractor
        )


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )

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
            batch_start = time.time()
            step += 1

            g_loss, d_loss = train_step(image_batch)
            print(
                f"Step {step}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Batch time: {time.time()-batch_start:.2f}s"
            )

        print(
            f"Epoch {epoch+1}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Time: {time.time()-start:.2f}s"
        )

        if (epoch + 1) % SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            generator.save("feature_gan_gen.keras")
            discriminator.save("feature_gan_disc.keras")


train(dataset, EPOCHS)
