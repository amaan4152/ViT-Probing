import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
)

from probe import Probe


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


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        size = [1, self.patch_size, self.patch_size, 1]
        stride = size
        patches = tf.image.extract_patches(
            images=images,
            sizes=size,
            strides=stride,
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.dense = Dense(units=projection_dims)
        self.pos_embed = Embedding(input_dim=num_patches, output_dim=projection_dims)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patches, delta=1)
        encode = self.dense(patch) + self.pos_embed(pos)
        return encode


class FullyConnected(layers.Layer):
	def __init__(self, widths, dropout):
		super(FullyConnected, self).__init__()
		self.FC = []
		for width in widths:
			self.FC.append(Dense(units=width, activation=tf.nn.gelu))
			self.FC.append(Dropout(dropout))

	def call(self, x):
		for fc_layer in self.FC:
			x = fc_layer(x)
		return x


class Preprocessor(layers.Layer):
    def __init__(self, num_patches, patch_size, projection_dims):

        super(Preprocessor, self).__init__()
        self.Patches = Patches(patch_size)
        self.PatchEncoder = PatchEncoder(num_patches, projection_dims)

    def call(self, x):
        x = self.Patches(x)
        x = self.PatchEncoder(x)
        return x


class VisionTransformer(layers.Layer):
    def __init__(
        self, num_encoders, num_heads, num_classes, projection_dims, insert_probes
    ):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.insert_probes = insert_probes
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.projection_dims = projection_dims
        self.transformer_units = [
            2 * projection_dims,
            projection_dims,
        ]

        self.Norm1 = LayerNormalization(epsilon=1e-6)
        self.Norm2 = LayerNormalization(epsilon=1e-6)
        self.Norm3 = LayerNormalization(epsilon=1e-6)

        self.AttentionHead = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.projection_dims, dropout=0.1
        )

        self.MLP_Encoder = FullyConnected(self.transformer_units, 0.1)
        self.GlobAvg1D = GlobalAveragePooling1D() 
        self.Flatten = Flatten()
        self.Dropout = Dropout(0.5)
        self.MLP_Head = FullyConnected([2048, 1024], 0.5)
        self.DenseClass = Dense(units=num_classes)

    def call(self, input):
        for id in range(self.num_encoders):
            x = self.Norm1(input)
            attention_out = self.AttentionHead(x, x)
            sum_1 = attention_out + input
            x = self.Norm2(x)
            x = self.MLP_Encoder(x)
            x += sum_1
            if self.insert_probes == True:
                x = Probe(self.num_classes, id)(x)

        x = self.Norm3(x)
        x = self.GlobAvg1D(x)
        x = self.Flatten(x)
        x = self.Dropout(x)
        x = self.MLP_Head(x)
        x = self.DenseClass(x)
        return x


"""
References:
(1) https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py
"""
