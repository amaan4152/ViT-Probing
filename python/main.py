import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import losses
from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.optimizers.schedules import PolynomialDecay, CosineDecay

# 3rd-Party scripts
from vit import DataAugmentation, Preprocessor, VisionTransformer
from CLI_parser import CLI_Parser

#   ----- MODEL CONFIGURATIONS -----
# get arguments from command-line -> see CLI_parser.py
ARGS = CLI_Parser()()

# training hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
BATCH_SIZE = 2048
EPOCHS = 100
LR_DECAY_TYPE = ARGS.LRDecay
LR_DECAY_STEPS = 1e4

# ViT characteristics
IMAGE_SIZE = 72
PATCH_SIZE = 4
PATCH_NUM = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECT_DIMS = 64
BOOL_PROBES = ARGS.probes
if "10" in ARGS.dataset:
    DATA = cifar10.load_data()
    NUM_CLASSES = 10
else:
    DATA = cifar100.load_data()
    NUM_CLASSES = 100


#   ----- MODEL SETUP -----
def vit_model(x_train, input_shape):
    inputs = Input(shape=input_shape)

    # augment data & perform mean-variance normalization
    augmentation = DataAugmentation(IMAGE_SIZE)
    augmentation.layers[0].adapt(x_train)
    x = augmentation(inputs)
    x = Preprocessor(
        num_patches=PATCH_NUM, patch_size=PATCH_SIZE, projection_dims=PROJECT_DIMS
    )(x)

    x = VisionTransformer(
        num_encoders=8,
        num_heads=4,
        num_classes=NUM_CLASSES,
        projection_dims=PROJECT_DIMS,
        insert_probes=BOOL_PROBES,
    )(x)

    return Model(inputs=inputs, outputs=x)


#  ----- GPU CONFIG -----
# alter GPU VRAM limit for handling large tensors (applies to small patch sizes)
# https://starriet.medium.com/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
def gpu_mem_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

#  ----- MODEL EXECUTION -----
def main():
    # get train/testz
    (train_data, train_labels), (test_data, test_labels) = DATA
    print('D0')
    model = vit_model(x_train=train_data, input_shape=(32, 32, 3))
    model.summary()

    # training
    lr_fn = LEARNING_RATE
    if LR_DECAY_TYPE == "linear":
        lr_fn = PolynomialDecay(LEARNING_RATE, LR_DECAY_STEPS, power=1)
    adam_w = tfa.optimizers.AdamW(learning_rate=lr_fn, weight_decay=WEIGHT_DECAY)
    model.compile(
        optimizer=adam_w,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
    )

    # fit
    model.fit(
        x=train_data,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
    )

    # evaluate
    _, acc, top_5_acc = model.evaluate(x=test_data, y=test_labels)
    print(f"Test accuracy: {acc.numpy():0.6f}")
    print(f"Test accuracy: {top_5_acc.numpy():0.6f}")

if __name__ == "__main__":
    if ARGS.kahan:
        gpu_mem_config()
    main()