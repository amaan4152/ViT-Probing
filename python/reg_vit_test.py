import colorama as color
from colorama import Fore
from colorama import Style

print(
    "["
    + Fore.YELLOW
    + Style.BRIGHT
    + "LOADING"
    + Style.RESET_ALL
    + "]: dependencies..."
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from os import getcwd

# 3rd-Party scripts
from CLI_parser import CLI_Parser
from trained_probe import train_probes
from vit_tf import VisionTransformer

print("[" + Fore.GREEN + Style.BRIGHT + "SUCCESS" + Style.RESET_ALL + "]: load")


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
PATCH_SIZE = 15
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


#   ----- MODEL SETUP ----- #
def vit_model(x_train, **kwargs):
    extra_layer = None
    if kwargs["add_conv"]:
        extra_layer = Conv2D(
            filters=16,
            kernel_size=int(3),
            activation="relu",
            padding="VALID",
        )

    model = VisionTransformer(
        x_train=x_train,
        image_size=IMAGE_SIZE,
        num_patches=PATCH_NUM,
        patch_size=PATCH_SIZE,
        num_encoders=NUM_ENCODERS,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        projection_dims=PROJECT_DIMS,
        test_layer=extra_layer,
    )
    return model


#  ----- MODEL EXECUTION ----- #
def train_model(*args, **kwargs):
    # get train/tests
    (train_data, train_labels), (test_data, test_labels) = DATA

    # init models
    model = vit_model(x_train=train_data, add_conv=kwargs["add_conv"])
    lr_sched = PolynomialDecay(power=1, initial_learning_rate=(8e-4), decay_steps=10000)
    adam = Adam(learning_rate=lr_sched)
    call_ES = EarlyStopping(patience=7)

    # compile model (no conv)
    model.compile(
        optimizer=adam,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.build(input_shape=(32, 32, 3))
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
    save_weights(model=model, config=kwargs["config"])

    # evaluate
    results = model.evaluate(x=test_data, y=test_labels)

    return H, results


def get_EncoderOutputs(add_conv):
    train, test = DATA
    model = vit_model(x_train=train[0], add_conv=add_conv)
    chkpt_dir = "chkpt-1" if add_conv else "chkpt-2"
    latest = tf.train.latest_checkpoint(f"checkpoints/tf/{chkpt_dir}/")
    model.load_weights(latest)
    model(train[0])
    x_train, y_train = model.outputs, train[1]
    model(test[0])
    x_test, y_test = model.outputs, test[1]
    return (x_train, y_train, x_test, y_test)


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
    while True:
        choice = input("Train [vit] or [probes]? ")
        if choice.lower() in ("vit", "probes"):
            break
        print(
            "["
            + Fore.LIGHTRED_EX
            + Style.BRIGHT
            + "ERROR"
            + Style.RESET_ALL
            + "]: Incorrect input, please retry..."
        )

    if choice == "vit":
        std_plot_name = input("Provide name of plot: ")
        H_conv, conv_results = train_model(
            add_conv=True, config="CONV"
        )  # saved in chkpt-1/
        H, results = train_model(add_conv=False, config="NOCONV")  # saved in chkpt-2/

        diff = conv_results[1] - results[1]
        plot_diagnostics(H.history, H_conv.history, std_plot_name)
        print(f"% diff: {100 * diff}")

    else:
        data = train_probes(get_EncoderOutputs(add_conv=False))
        conv_data = train_probes(get_EncoderOutputs(add_conv=True))
        plt.figure()
        plt.suptitle(f"CIFAR-10 Probes ViT: S_{PATCH_SIZE}-RES_{IMAGE_SIZE}")
        plt.xlabel("Probe #")
        plt.xticks(data["x"])
        plt.ylim((0, 80))
        plt.ylabel("Accuracy [%]")
        plt.plot(data["x"], data["y"], color="orange", label="No conv")
        plt.plot(
            conv_data["x"], conv_data["y"], "--", color="orange", label="With conv"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig("trained_probes.png")


# ----- UTILITY FUNCTIONS ----- #
# https://stackoverflow.com/questions/21716940/is-there-a-way-to-track-the-number-of-times-a-function-is-called/21717084
"""
    Custom decorator function to generate new directories for each saved checkpoint set
    chkpt-# => dir of every associated save_weights call
"""


def diradjust(fn):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        path = f"{WRK_DIR}/checkpoints/tf/chkpt-{wrapper.calls}/S_{PATCH_SIZE}-RES_{IMAGE_SIZE}-{kwargs['config']}"
        print(f"=== NEW CHECKPOINT ADDED: chkpt-{wrapper.calls} ===")
        return fn(model=kwargs["model"], path=path)

    wrapper.calls = 0
    return wrapper


@diradjust
def save_weights(*args, **kwargs):
    kwargs["model"].save_weights(kwargs["path"])


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


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    if ARGS.kahan:
        gpu_mem_config()

    main()
