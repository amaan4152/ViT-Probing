import timm
import torch

import sys
import tqdm
import numpy as np
import einops as e

import tensorflow as tf
import tensorflow_datasets as tfds

RESOLUTION = 224
BATCH_SIZE = 512
NUM_PROBES = 12
EPOCH_PROBES = 100
DATASET = "cifar10"
NUM_TRAIN = 50000
NUM_TEST = 10000

# https://colab.research.google.com/github/google-research/vision_transformer/blob/master/vit_jax_augreg.ipynb#scrollTo=gd6NVWNLYOSI


def pp(img, sz):
    """Simple image preprocessing."""
    img = tf.cast(img, float) / 255.0
    img = tf.image.resize(img, [sz, sz])
    return img


def pp_torch(img, sz):
    """Simple image preprocessing for PyTorch."""
    img = pp(img, sz)
    img = img.numpy().transpose([0, 3, 1, 2])  # PyTorch expects NCHW format.
    return torch.tensor(img[None])


def get_EncoderOutputs():
    # get vit with timm and associated pretrained weights
    model = timm.create_model("vit_small_patch32_224", num_classes=10)
    timm.models.load_checkpoint(model, "vit_s32.npz")

    """
        Register a hook function to the model to extract
        encoder outputs. 

        Each block represents an encoder block.
    """
    encoder_features = []
    # https://pytorch.org/blog/FX-feature-extraction-torchvision/
    def hook_feat_map(mod, inp, out):
        encoder_features.append(out.numpy())

    for block in model.blocks:
        block.register_forward_hook(hook_feat_map)

    model.eval()

    # init train/test data variables and CIFAR10 dataset
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    ds = tfds.load(DATASET, batch_size=BATCH_SIZE)
    # use arbitrarily large shuffle buffer for complete shuffling
    ds["train"] = ds["train"].shuffle(NUM_TRAIN // 2)
    ds["test"] = ds["test"].shuffle(NUM_TEST // 2)

    """
        1) Capture a subset of training/test CIFAR10 data 
        2) Feed it through the model
        3) Add encoder outputs to train/test list
        4) Clear and delete temp vars; clear GPU cache; detach PyTorch Tensors
            - Prevent PyTorch graph history from accumulating and cause
              Out-Of-Memory (OOM) runtime errors
            - Delete large variables to prevent (OOM)
    """
    try:
        with torch.no_grad():
            print("=== TRAIN DATA ACQUISITION -> [START] ===")
            for batch, _ in zip(ds["train"], tqdm.trange(10)):
                model(pp_torch(batch["image"], RESOLUTION)[0]).detach()
                # make sure batch size dimension is consistent
                if encoder_features[0].shape[0] != BATCH_SIZE:
                    print("NOT SAME SIZE")
                    del encoder_features, batch
                    encoder_features = []
                    torch.cuda.empty_cache()
                    continue

                x_train.append(np.array(encoder_features))
                y_train.extend(batch["label"].numpy())

                del encoder_features, batch
                encoder_features = []
                torch.cuda.empty_cache()

            print("=== TRAIN DATA ACQUISITION -> [END] ===")
            print("=== TEST DATA ACQUISITION -> [START] ===")
            encoder_features = []
            for batch, _ in zip(ds["test"], tqdm.trange(10)):
                model(pp_torch(batch["image"], RESOLUTION)[0]).detach()
                if encoder_features[0].shape[0] != BATCH_SIZE:
                    print("NOT SAME SIZE")
                    del encoder_features, batch
                    encoder_features = []
                    torch.cuda.empty_cache()
                    continue

                x_test.append(np.array(encoder_features))
                y_test.extend(batch["label"].numpy())

                del encoder_features, batch
                encoder_features = []
                torch.cuda.empty_cache()

            print("=== TEST DATA ACQUISITION -> [END] ===")

    except RuntimeError:
        print(sys.exc_info())
        exit(1)
    del model, ds

    """
        1) Convert trianing/test data into numpy ndarrays for processing
        2) Rearrange dimensions for probe analysis
        3) Catch any runtime errors -> primarily to prevent OOM events
    """
    try:
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        # rearrange dimensions:
        # from: dims(NUM_EPOCHS, NUM_ENCODERS, BATCH_SIZE, CHANNEL, HEIGHT)
        # to:   dims(NUM_ENCODERS, NUM_IMAGES, HEIGHT, CHANNEL)
        x_train = e.rearrange(x_train, "e n b c h -> n (e b) h c")
        x_test = e.rearrange(x_test, "e n b c h -> n (e b) h c")
    except RuntimeError:
        print(sys.exc_info())
        exit(1)

    return x_train, y_train, x_test, y_test
