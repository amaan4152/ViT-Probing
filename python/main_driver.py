from utility import *
printWarning("loading dependencies ...")

# dependencies
import colorama as color
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple

# 3rd-Party scripts
from CLI_parser import CLI_Parser
from conv_tf import PatchConv
from train_probes import train_probes
from vit_tf import VisionTransformer

printGood("dependencies loaded!")


# ----- CONFIGURATIONS ----- #
# get arguments from command-line -> see CLI_parser.py
ARGS = CLI_Parser()()

#   ----- MODEL CONFIGURATIONS ----- #
# training hyperparameters
BATCH_SIZE = 4096
EPOCHS = 100

# ViT hyperparameters
IMAGE_SIZE = 32
PATCH_SIZE = 15
PATCH_NUM = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECT_DIMS = 36
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
def vit_model(x_train, **kwargs) -> Model:
    printWarning("ViT Model")
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


def pconv_model(x_train, **kwargs) -> Model:
    printWarning("PConv Model")
    extra_layer = None
    p_size = PATCH_SIZE + 1
    if kwargs["add_conv"]:
        p_size -= 1
        extra_layer = Conv2D(
            filters=16,
            kernel_size=int(3),
            activation="relu",
            padding="VALID",
        )

    model = PatchConv(
        x_train=x_train,
        image_size=IMAGE_SIZE,
        patch_size=p_size,
        num_classes=NUM_CLASSES,
        test_layer=extra_layer,
    )
    return model


getModel = {"vit": vit_model, "pconv": pconv_model}

#  ----- MODEL EXECUTION ----- #
def train_model(*args, **kwargs) -> Tuple[History, List]:
    # get train/tests
    (train_data, train_labels), (test_data, test_labels) = DATA

    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        # init models
        model = getModel[kwargs["model_type"]](
            x_train=train_data, add_conv=kwargs["add_conv"]
        )

        # learning rate scheduler
        lr_sched = PolynomialDecay(
            power=tf.Variable(1),
            initial_learning_rate=tf.Variable(8e-4),
            decay_steps=tf.Variable(10000),
        )

        # https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
        # Adam optimizer
        adam = Adam(
            learning_rate=lr_sched,
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
        )
        call_ES = EarlyStopping(patience=tf.Variable(7))

        # compile model (no conv)
        model.compile(
            optimizer=adam,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )   
        model.summary_model()

    # fit
    H = model.fit(
        x=train_data,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[call_ES],
    )

    # save trained weights for training probes
    save_weights(model=model, model_type=kwargs["model_type"], config=kwargs["config"])

    # evaluate
    results = model.evaluate(x=test_data, y=test_labels)

    return H, results


def get_EncoderOutputs(add_conv, model_type):
    """
    Compose train and test data for probes based on output of selected layers
    established inside the ViT model class
    """
    tf.keras.backend.clear_session()
    train, test = DATA
    addConv_str = "CONV" if add_conv else "NO-CONV"
    printWarning(f"{model_type} | {addConv_str} ~~~~~~~~~~~~~")
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        model = getModel[model_type](x_train=train[0], add_conv=add_conv)

        # get latest checkpoint weights for both model types
        config = "conv" if add_conv else "std"
        chkpt_dir = f"chkpt-{model_type}-{config}"
        print(f"=== CHKPT DIR {chkpt_dir} ===")
        latest = tf.train.latest_checkpoint(f"checkpoints/tf/{chkpt_dir}/")
        model.load_weights(latest)

    # generate train dataset
    x_train = train[0]
    print(x_train.shape)
    model(x_train)
    x_train, y_train = model.outputs, train[1]

    # generate test dataset
    model(test[0])
    x_test, y_test = model.outputs, test[1]

    return (x_train, y_train, x_test, y_test)


def main() -> int:
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
    # override prompts with slurm inputs
    if ARGS.slurm == "":
        while True:
            choice = input("Train [vit/pconv] or [probes]? ").lower()
            if choice in ("vit", "pconv", "probes"):
                break

            printError("Invalid input sequence, try again!")

        std_plot_name = input("Provide name of plot: ")
    else:
        choice, std_plot_name = ARGS.slurm.split()
    
    if choice in getModel.keys():
        H_conv, conv_results = train_model(
            add_conv=True, config="CONV", model_type=choice
        )  # saved in chkpt-{choice}-conv/
        H, results = train_model(
            add_conv=False, config="NOCONV", model_type=choice
        )  # saved in chkpt-{choice}-std/
        
        diff = conv_results[1] - results[1]
        suptitle = f'{choice}: S_{PATCH_SIZE}-RES_{IMAGE_SIZE}'
        plot_diagnostics(H.history, H_conv.history, std_plot_name, suptitle)
        print(f"% diff: {100 * diff}")
    else:
        # init figure
        fig = plt.figure(figsize=(19, 11))
        plt_shape = [1, 2, 1]
        
        for m_type in ('pconv', 'vit'):
            # label preparation
            no_conv_labels = [
                "Input",
                "Prep",
            ]
            conv_labels = ["Input", "PreConv", "Prep"]
            num_hidden = NUM_ENCODERS
            hidden_type = "Enc"
            if m_type == "pconv":
                hidden_type = "Conv2D"
                num_hidden = 1

            for i in range(num_hidden):
                no_conv_labels.append(f"{hidden_type}{i}")
                conv_labels.append(f"{hidden_type}{i}")

            # setup axes handler
            ax = fig.add_subplot(*plt_shape)
            ax.set_title(f"{m_type.upper()}")
            ax.set_xlabel("Probe #")
            ax.set_yscale("function", functions=(forward, inverse))
            ax.set_ylabel("Accuracy [%]")

            # get probe data
            conv_data = train_probes(get_EncoderOutputs(
                add_conv=True,
                model_type=m_type
            ))
            std_data = train_probes(get_EncoderOutputs(
                add_conv=False,
                model_type=m_type
            ))
            
            # mercator y-axis scaling
            max_y = np.max([np.max(std_data["y"]), np.max(conv_data["y"])])
            ax.yaxis.set_major_locator(FixedLocator(np.arange(0, max_y + 5) ** 2))
            ax.yaxis.set_major_locator(FixedLocator(np.arange(0, max_y + 5)))
            ax.set_xticks(conv_data["x"])
            ax.set_xticklabels(conv_labels, rotation=45)

            # plot probe data
            ax.scatter(std_data["x"], std_data["y"], marker="x", label="No conv")
            ax.scatter(conv_data["x"], conv_data["y"], marker="o", label="With conv")
            
            plt_shape = [1, 2, 2]
            ax.grid()
            ax.legend(loc="lower right")

        # tidy up plot
        plt.tight_layout()
        plt.show()
        plt.savefig(f"probe_plots/{std_plot_name}")

        return 0


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    if ARGS.kahan:
        gpu_mem_config()

    main()
