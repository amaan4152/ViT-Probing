import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import cifar10, cifar100

# 3rd-Party scripts
from vit import DataAugmentation, Preprocessor, VisionTransformer
from CLI_parser import CLI_Parser

#   ----- MODEL CONFIGURATIONS -----
# get arguments from command-line -> see CLI_parser.py
ARGS = CLI_Parser()()

# training hyperparameters
BATCH_SIZE = 1024
EPOCHS = 100

# ViT hyperparameters
IMAGE_SIZE = 32
PATCH_SIZE = 16
PATCH_NUM = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECT_DIMS = 384
NUM_ENCODERS = 12
NUM_HEADS = 6
BOOL_PROBES = ARGS.probes
if "10" in ARGS.dataset:  # CIFAR-10
    DATA = cifar10.load_data()
    NUM_CLASSES = 10
else:  # CIFAR-100
    DATA = cifar100.load_data()
    NUM_CLASSES = 100


#  ----- GPU CONFIG -----
# alter GPU VRAM limit for handling large tensors (applies to small patch sizes)
# https://starriet.medium.com/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
def gpu_mem_config():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


#   ----- MODEL SETUP -----
def vit_model(x_train, input_shape):
    input = Input(shape=input_shape)

    # augment data & perform mean-variance normalization
    augmentation = DataAugmentation()
    augmentation.layers[0].adapt(x_train)
    x = augmentation(input)
    x = Preprocessor(
        num_patches=PATCH_NUM, patch_size=PATCH_SIZE, projection_dims=PROJECT_DIMS
    )(x)

    x = VisionTransformer(
        num_encoders=NUM_ENCODERS,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        projection_dims=PROJECT_DIMS,
        insert_probes=BOOL_PROBES,
    )(x)
    output = x
    return Model(inputs=input, outputs=output)


#  ----- MODEL EXECUTION -----
def main():
    # get train/tests
    (train_data, train_labels), (test_data, test_labels) = DATA

    model = vit_model(x_train=train_data, input_shape=(32, 32, 3))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    """
    Custom weight initializer using pretrained weights.
    Bit buggy implementation; embedding dims are not compliant


    for layer in model.layers[2:]:
        layer.set_weights()
    """

    # fit
    model.fit(
        x=train_data,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
    )

    # evaluate
    _, acc = model.evaluate(x=test_data, y=test_labels)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    if ARGS.kahan:
        gpu_mem_config()

    main()
