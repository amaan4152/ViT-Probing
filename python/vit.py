import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    Flatten,
)

from untrained_probe import Probe
from checkpoint_loader import load_weights

WEIGHTS_FILE = np.load("vit-S16-fine.npz")


def DataAugmentation():
    data_augmentation = Sequential(
        [
            layers.Normalization(),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    return data_augmentation


# for testing
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        N = (images.shape[1] // self.patch_size) ** 2
        return tf.reshape(patches, shape = (batch_size, self.patch_size, self.patch_size, patch_dims))
        """
        return images


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()
        self.projection_dims = projection_dims
        self.num_patches = num_patches

        self.patch_embed = Conv2D(filters=projection_dims, kernel_size=16, strides=16)
        self.class_embed = self.add_weight(
            name="CLS", shape=(1, 1, projection_dims), initializer="uniform"
        )

        self.pos_embed = Embedding(input_dim=num_patches, output_dim=projection_dims)

    def set_weights(self):
        patch_embed_weights = load_weights(WEIGHTS_FILE, "embedding/")
        class_embed_weights = load_weights(WEIGHTS_FILE, "cls")
        posit_embed_weights = load_weights(WEIGHTS_FILE, "pos_embedding")

        self.patch_embed.set_weights(patch_embed_weights[::-1])
        self.class_embed.assign(class_embed_weights[0])
        self.pos_embed.set_weights(posit_embed_weights)

    def call(self, patch):
        patch_embed = self.patch_embed(patch)
        patch_embed = tf.reshape(
            patch_embed,
            (
                tf.shape(patch_embed)[0],
                tf.shape(patch_embed)[1] * tf.shape(patch_embed)[2],
                tf.shape(patch_embed)[3],
            ),
        )

        cls_embed = tf.broadcast_to(
            self.class_embed, [tf.shape(patch)[0], 1, self.projection_dims]
        )
        patch_embed = tf.concat([cls_embed, patch_embed], axis=1)

        self.pos_embed = self.add_weight(
            name="POS_EMBED", shape=(1, patch_embed.shape[1], patch_embed.shape[2])
        )
        encode = patch_embed + self.pos_embed
        return encode


class Preprocessor(layers.Layer):
    def __init__(self, num_patches, patch_size, projection_dims):
        super(Preprocessor, self).__init__()
        self.Patches = Patches(patch_size)
        self.PatchEncoder = PatchEncoder(num_patches, projection_dims)

    def set_weights(self):
        self.PatchEncoder.set_weights()

    def call(self, x):
        x = self.Patches(x)
        x = self.PatchEncoder(x)
        return x


class FullyConnected(layers.Layer):
    def __init__(self, widths, dropout):
        super(FullyConnected, self).__init__()
        self.FC = []
        for width in widths:
            self.FC.append(
                Dense(units=width, activation=tf.nn.gelu, name="dense_" + str(width))
            )
            self.FC.append(Dropout(dropout, name="dropout_" + str(width)))

    def set_weights(self, encoder_id):
        id = 0
        for dense in self.FC:
            if "dropout" in dense.name:
                continue
            dense_weights = load_weights(WEIGHTS_FILE, encoder_id, f"Dense_{id}")
            dense.set_weights(dense_weights[::-1])
            id += 1

    def call(self, x):
        for fc_layer in self.FC:
            x = fc_layer(x)
        return x


class VisionTransformer(layers.Layer):
    def __init__(
        self, num_encoders, num_heads, num_classes, projection_dims, insert_probes
    ):
        super(VisionTransformer, self).__init__()

        self.probes_out = [0] * num_encoders

        self.num_classes = num_classes
        self.insert_probes = insert_probes
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.projection_dims = projection_dims
        self.transformer_units = [
            4 * projection_dims,
            projection_dims,
        ]

        self.Norm1 = LayerNormalization(epsilon=1e-6)
        self.Norm2 = LayerNormalization(epsilon=1e-6)
        self.Norm3 = LayerNormalization(epsilon=1e-6)

        self.AttentionHead = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=int(self.projection_dims / 6), dropout=0.1
        )

        self.MLP_Encoder = FullyConnected(self.transformer_units, 0.1)
        self.Flatten = Flatten()
        self.Dropout = Dropout(0.5)
        self.Head = Dense(units=num_classes)

    def set_weights(self):
        for id in range(self.num_encoders):
            encoder_id = "encoderblock_" + str(id) + "/"
            norm1_weights = load_weights(WEIGHTS_FILE, encoder_id, "LayerNorm_0")
            norm2_weights = load_weights(WEIGHTS_FILE, encoder_id, "LayerNorm_2")
            MSA_weights_k = load_weights(
                WEIGHTS_FILE, encoder_id, "MultiHeadDotProductAttention", "key"
            )
            MSA_weights_q = load_weights(
                WEIGHTS_FILE, encoder_id, "MultiHeadDotProductAttention", "query"
            )
            MSA_weights_v = load_weights(
                WEIGHTS_FILE, encoder_id, "MultiHeadDotProductAttention", "value"
            )
            MSA_weights_o = load_weights(
                WEIGHTS_FILE, encoder_id, "MultiHeadDotProductAttention", "out"
            )
            MSA_net_weights = [
                *MSA_weights_k[::-1],
                *MSA_weights_q[::-1],
                *MSA_weights_v[::-1],
                *MSA_weights_o[::-1],
            ]

            self.Norm1.set_weights(norm1_weights[::-1])
            self.Norm2.set_weights(norm2_weights[::-1])
            self.AttentionHead.set_weights(MSA_net_weights)
            self.MLP_Encoder.set_weights(encoder_id)

        norm3_weights = load_weights(WEIGHTS_FILE, "encoder_norm")
        head_weights = load_weights(WEIGHTS_FILE, "head")

        self.Norm3.set_weights(norm3_weights[::-1])
        self.Head.set_weights(head_weights[::-1])

    def call(self, input, training=False):
        for id in range(self.num_encoders):
            x = self.Norm1(input)
            attention_out = self.AttentionHead(x, x)
            sum_1 = attention_out + input
            x = self.Norm2(x)
            x = self.MLP_Encoder(x)
            x += sum_1
            if self.insert_probes:
                tf.stop_gradient(Probe(self.num_classes, id)(x))
            input = x

        x = self.Norm3(x)
        x = self.Head(x[:, 0])

        if self.insert_probes:
            return (x, tf.convert_to_tensor(self.probes_out, dtype=tf.float32))
        return x


"""
References:
(1) https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py
"""
