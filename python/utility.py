from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

WRK_DIR = getcwd()
IMAGE_SIZE = 32
PATCH_SIZE = 16

# ----- UTILITY FUNCTIONS ----- #
def forward(y):
    """https://matplotlib.org/stable/gallery/scales/scales.html#sphx-glr-gallery-scales-scales-py"""
    y = np.deg2rad(y)
    return np.rad2deg(np.log(np.abs(np.tan(y) + 1.0 / np.cos(y))))


def inverse(y):
    """https://matplotlib.org/stable/gallery/scales/scales.html#sphx-glr-gallery-scales-scales-py"""
    y = np.deg2rad(y)
    return np.rad2deg(np.arctan(np.sinh(y)))


def diradjust(fn):
    """
    Custom decorator function to generate new directories for each saved checkpoint set
    chkpt-# => dir of every associated save_weights call

    Citation: https://stackoverflow.com/questions/21716940/is-there-a-way-to-track-the-number-of-times-a-function-is-called/21717084
    """

    def wrapper(*args, **kwargs):
        if kwargs["config"] == "NOCONV":
            config = "-std"

        elif kwargs["config"] == "CONV":
            config = "-conv"

        path = f"{WRK_DIR}/checkpoints/tf/chkpt-{kwargs['model_type']}{config}/S_{PATCH_SIZE}-RES_{IMAGE_SIZE}-{kwargs['config']}"
        print(f"=== NEW CHECKPOINT ADDED: chkpt-{kwargs['model_type']}{config} ===")
        return fn(model=kwargs["model"], path=path)

    return wrapper


@diradjust
def save_weights(*args, **kwargs):
    kwargs["model"].save_weights(kwargs["path"])


def gpu_mem_config():
    """
    Alter GPU VRAM limit for handling large tensors (applies to small patch sizes)

    Citation: https://starriet.medium.com/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
    """
    gpus = list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plot_diagnostics(history, history_with_conv, plot_name):
    """
    Plot accuracy and loss diagnostics of ViT with convolution and without convolution
    """
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
