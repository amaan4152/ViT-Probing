import colorama as color
from colorama import Fore
from colorama import Style
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from os import getcwd

# 3rd-Party scripts
from vit_tf import DataAugmentation, VisionTransformer
from CLI_parser import CLI_Parser

# ----- CONFIGURATIONS ----- #
# get arguments from command-line -> see CLI_parser.py
WRK_DIR = getcwd()
ARGS = CLI_Parser()()

#   ----- MODEL CONFIGURATIONS ----- #
# training hyperparameters
BATCH_SIZE = 1024
EPOCHS = 100

# ViT hyperparameters
IMAGE_SIZE = 32
PATCH_SIZE = 8
PATCH_NUM = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECT_DIMS = 32
NUM_ENCODERS = 12
NUM_HEADS = 6
BOOL_PROBES = ARGS.probes
if "10" in ARGS.dataset:  # CIFAR-10
    DATA = cifar10.load_data()
    NUM_CLASSES = 10
else:  # CIFAR-100
    DATA = cifar100.load_data()
    NUM_CLASSES = 100


# plot loss/accuracy history
def plot_diagnostics(history, history_with_conv, plot_name):
    # plot loss
    plt.subplot(211)
    plt.title("Loss")
    plt.plot(history["loss"], color="blue", label="Train")
    plt.plot(history["val_loss"], color="red", label="Validation")
    plt.plot(history_with_conv["loss"], "--", color="blue", label="Train (with conv)")
    plt.plot(
        history_with_conv["val_loss"], "--", color="red", label="Validation (with conv)"
    )

    # plot accuracy
    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(history["accuracy"], color="blue", label="Train")
    plt.plot(history["val_accuracy"], color="red", label="Validation")
    plt.plot(
        history_with_conv["accuracy"], "--", color="blue", label="Train (with conv)"
    )
    plt.plot(
        history_with_conv["val_accuracy"],
        "--",
        color="red",
        label="Validation (with conv)",
    )

    plt.suptitle(f"ViT: S_{PATCH_SIZE}-RES_{IMAGE_SIZE}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{WRK_DIR}/plots/{plot_name}.png")


#  ----- GPU CONFIG ----- #
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


#   ----- MODEL SETUP ----- #
def vit_model(x_train, input_shape, add_conv=False):
    input = Input(shape=input_shape)

    # augment data & perform mean-variance normalization
    augmentation = DataAugmentation(IMAGE_SIZE)
    augmentation.layers[0].adapt(x_train)
    x = augmentation(input)

    if add_conv:
        x = Conv2D(
            filters=PATCH_SIZE,
            kernel_size=int(PATCH_SIZE * 2),
            activation="relu",
            padding="SAME",
        )(x)

    x = VisionTransformer(
        num_patches=PATCH_NUM,
        patch_size=PATCH_SIZE,
        num_encoders=NUM_ENCODERS,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        projection_dims=PROJECT_DIMS,
    )(x)
    output = x
    return Model(inputs=input, outputs=output)


#  ----- MODEL EXECUTION ----- #
def train_model(*args):
    # get train/tests
    (train_data, train_labels), (test_data, test_labels) = DATA

    # init models
    model = vit_model(x_train=train_data, input_shape=(32, 32, 3))
    model_with_conv = vit_model(
        x_train=train_data, input_shape=(32, 32, 3), add_conv=True
    )
    lr_sched = PolynomialDecay(power=1, initial_learning_rate=(8e-4), decay_steps=10000)
    adam = Adam(learning_rate=lr_sched)
    call_ES = EarlyStopping(patience=7)

    # compile model (with conv)
    model_with_conv.compile(
        optimizer=adam,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model_with_conv.summary()

    # fit model (with conv)
    H_with_conv = model_with_conv.fit(
        x=train_data,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[call_ES],
    )

    # save trained weights for training probes
    model_with_conv.save_weights(
        f"{WRK_DIR}/checkpoints/tf/S_{PATCH_SIZE}-RES_{IMAGE_SIZE}-CONV_F{2*PATCH_SIZE}_K{PATCH_SIZE}"
    )

    # compile model (no conv)
    model.compile(
        optimizer=adam,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    # fit (no conv)
    H = model.fit(
        x=train_data,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[call_ES],
    )

    # save trained weights for training probes
    model.save_weights(
        f"{WRK_DIR}/checkpoints/tf/S_{PATCH_SIZE}-RES_{IMAGE_SIZE}-NOCONV"
    )

    # evaluate
    results = model.evaluate(x=test_data, y=test_labels)
    results_with_conv = model_with_conv.evaluate(x=test_data, y=test_labels)

    # plot loss & accuracies
    plot_diagnostics(H.history, H_with_conv.history, args[0])

    diff = results_with_conv[1] - results[1]
    print(f"% diff: {100 * diff}")

def get_EncoderOutputs():
    # get train/tests
    (train_data, train_labels), (test_data, test_labels) = DATA 
    model = vit_model(x_train=train_data, input_shape=(32, 32, 3))
    print(model.layers)

def main():
    color.init()
    print(
        "===== "
        + Fore.CYAN
        + Style.BRIGHT
        + "Vision Transformer"
        + Style.RESET_ALL
        + " : "
        + Fore.MAGENTA
        + Style.BRIGHT
        + "Linear Probing"
        + Style.RESET_ALL
        + " ====="
    )
    std_plot_name = input("Provide name of plot: ")
    train_model(std_plot_name)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    if ARGS.kahan:
        gpu_mem_config()

    main()
