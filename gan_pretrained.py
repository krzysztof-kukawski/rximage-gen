import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from rotator import image_generator
from keras import saving

# Hyperparameters
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
    for fname in os.listdir(images_120)[:5000]
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
    model = saving.load_model("gan_gen.keras")
    return model


def make_discriminator_model():
    model = saving.load_model("gan_disc.keras")
    return model


generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0009, beta_1=0.5, beta_2=0.999
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0009, beta_1=0.5, beta_2=0.999
)
def add_instance_noise(images, stddev=0.02):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=stddev)
    return images + noise


@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]

    noise = tf.random.normal([batch_size, NOISE_DIM])
    noise += tf.random.normal(shape=noise.shape, mean=0.0, stddev=0.05)
    real_labels = tf.ones((batch_size, 1)) * 0.9  # Label smoothing
    fake_labels = tf.zeros((batch_size, 1))

    if tf.random.uniform(()) < 0.05:
        real_labels, fake_labels = fake_labels, real_labels

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_images_noisy = add_instance_noise(real_images)
        fake_images_noisy = add_instance_noise(generated_images)

        real_output = discriminator(real_images_noisy, training=True)
        fake_output = discriminator(fake_images_noisy, training=True)

        d_loss_real = cross_entropy(real_labels, real_output)
        d_loss_fake = cross_entropy(fake_labels, fake_output)
        d_loss = d_loss_real + d_loss_fake

    gradients_of_discriminator = disc_tape.gradient(
        d_loss, discriminator.trainable_variables
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    noise = tf.random.normal([batch_size, NOISE_DIM])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        g_loss = cross_entropy(tf.ones((batch_size, 1)), fake_output)

    gradients_of_generator = gen_tape.gradient(
        g_loss, generator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )

    return g_loss, d_loss



def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)
    predictions = (predictions + 1.0) / 2.0  # Scale from [-1, 1] to [0, 1]

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
            generator.save("gan_gen.keras")
            discriminator.save("gan_disc.keras")


train(dataset, EPOCHS)
