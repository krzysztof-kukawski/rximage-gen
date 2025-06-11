import os
import time
from keras import Model
from keras import layers, saving
import tensorflow as tf
import matplotlib.pyplot as plt
from rotator import image_generator
os.chdir(os.path.dirname(__file__))

gpus = tf.config.list_physical_devices('GPU')
tf.config.optimizer.set_jit(True)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
    except RuntimeError as e:
        print(e)
NOISE_DIM = 100
BATCH_SIZE = 42
EPOCHS = 1000
SAVE_INTERVAL = 5
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
        lambda: image_generator(image_paths, angles=[180, 90, 270], target_size=(90, 120)),
        output_signature=tf.TensorSpec(shape=(90, 120, 3), dtype=tf.float16),
    )
    batched_dataset = dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return batched_dataset


dataset = load_dataset()


def make_generator_model() -> Model:
    try:
        model: Model = saving.load_model("feature_gan_gen1.keras") # type: ignore
    except ValueError:
        model: Model = saving.load_model("feature_gan_gen2.keras") # type: ignore
        
    return model


def make_discriminator_model() -> tuple[Model, list[Model]]:
    
    feature_extractors: list[Model] = []
    try:
        model: Model = saving.load_model("feature_gan_disc1.keras") # type: ignore
    except ValueError:
        model: Model = saving.load_model("feature_gan_disc2.keras") # type: ignore
        
    intermediate_layer_names = ["leaky_re_lu_1","leaky_re_lu_3", "leaky_re_lu_4"]
    for intermediate_layer_name in intermediate_layer_names:
        feature_layer = model.get_layer(intermediate_layer_name).output
        feature_extractor = Model(inputs=model.input, outputs=feature_layer)
        feature_extractors.append(feature_extractor)

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
    
    λ = tf.minimum(2.0, epoch / 10.0)
    total_loss = adv_loss + λ * fm_loss
    return total_loss


discriminator, feature_extractors = make_discriminator_model()
print(discriminator.summary())
generator = make_generator_model()
print(generator.summary())
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.5, beta_2=0.999
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.5, beta_2=0.999
)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images, epoch):
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

    for i, img in enumerate(predictions[:18]):
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
                f"Step {step}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Batch time: {time.time()-batch_start:.5f}s"
            )

        print(
            f"Epoch {epoch+1}, Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}, Time: {time.time()-start:.2f}s"
        )

        if (epoch + 1) % SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            generator.save(f"feature_gan_gen{s}.keras", overwrite=True)
            discriminator.save(f"feature_gan_disc{s}.keras", overwrite=True)
            s +=1
            if s == 3:
                s = 1
            

train(dataset, EPOCHS)
