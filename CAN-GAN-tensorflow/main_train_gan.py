import tensorflow as tf
import datetime, os, time, sys
from tensorflow.keras.layers import *

sys.path.insert(0, os.getcwd())
from tools.options import Options
import tools.utils_gan as utils
import net_can_gan as net

# five inputs

LAMBDA = 100
BUFFER_SIZE = 500
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():
    def generator_mae(gen_output, target):
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        return l1_loss

    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = loss_object(
            tf.ones_like(disc_generated_output), disc_generated_output
        )
        # mean absolute error
        l1_loss = generator_mae(gen_output, target)
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(
            tf.zeros_like(disc_generated_output), disc_generated_output
        )
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    @tf.function
    def validation_step(input_image, target, epoch):
        gen_output = generator(input_image, training=False)
        gen_validation_total_loss = generator_mae(gen_output, target)
        with summary_writer.as_default():
            tf.summary.scalar(
                "gen_validation_total_loss", gen_validation_total_loss, step=epoch
            )

    @tf.function
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator(
                [input_image, gen_output], training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=epoch)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=epoch)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=epoch)
            tf.summary.scalar("disc_loss", disc_loss, step=epoch)

    def fit(train_ds, validate_ds, epochs,output_dir):
        for epoch in range(epochs):
            start = time.time()
            if args.viz:
                for example_input, example_target in validate_ds.take(1):
                    utils.generate_images(generator, example_input, example_target)
            print("Epoch: ", epoch)

            # Train
            for n, (input_image, target) in train_ds.enumerate():
                print(".", end="")
                if (n + 1) % 100 == 0:
                    print()
                # target =tf.image.rgb_to_grayscale(target)
                train_step(input_image, target, epoch)
            print()

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                for n, (input_image, target) in validate_ds.enumerate():
                    if n == 0:
                        filename_output = "%s/Current.png" % (output_dir)
                        utils.save_images0(
                            generator,
                            input_image,
                            filename_output,
                        )
                    validation_step(
                        input_image,
                        target,
                        epoch,
                    )

            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, time.time() - start
                )
            )
        
        checkpoint.save(file_prefix=checkpoint_prefix)


    args = Options().parse()
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    epochs = args.epochs
    BATCH_SIZE = args.batch_size
    ## Directory of the Dataset
    print("---> Loading Datasets...")
    # tf.debugging.set_log_device_placement(True)

    train_dataset = tf.data.Dataset.list_files(args.train_set + "/*.png")
    train_dataset = train_dataset.map(
        utils.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.list_files(args.val_set + "/*.png")
    val_dataset = val_dataset.map(utils.load_image_val)
    val_dataset = val_dataset.batch(1)

    log_dir = args.log_dir

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    print("---> Constructing Network Architecture...")
    generator = net.build_CAN()
    tf.keras.utils.plot_model(generator, show_shapes=True)
    discriminator = net.Discriminator()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    checkpoint.restore(tf.train.latest_checkpoint(args.resume_dir))
    output_dir = args.checkpoint_dir
    fit(train_dataset, val_dataset, epochs, output_dir)


if __name__ == "__main__":
    main()
