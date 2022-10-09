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

    args = Options().parse()
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    ## Directory of the Dataset
    print("---> Loading Datasets...")
    # tf.debugging.set_log_device_placement(True)
    test_dataset = tf.data.Dataset.list_files(args.test_set + "/*.png")
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    print("---> Constructing Network Architecture...")
    generator = net.build_CAN()
    tf.keras.utils.plot_model(generator, show_shapes=True)
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt-215")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_prefix)
    for n, filename in test_dataset.enumerate():
        input_image, imSize = utils.load_image_test(filename)
        filename_output = "%s/%06d.png" % (args.output_dir, n + 1)
        # start = time.time()
        utils.save_images(
            generator,
            input_image[tf.newaxis, ...],
            imSize,
            filename_output,
        )
            

if __name__ == "__main__":
    main()
