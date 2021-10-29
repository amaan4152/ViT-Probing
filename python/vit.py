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
from keras import backend as K

from probe import Probe

def DataAugmentation(image_size):
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
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        N = (images.shape[1] // self.patch_size) ** 2
        return tf.reshape(patches, shape = (batch_size, N, patch_dims))


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()

        self.projection_dims = projection_dims
        self.num_patches = num_patches

        self.dense = Dense(units=projection_dims)
        self.class_embed = self.add_weight(name='CLS', shape = (1, 1, projection_dims), initializer='uniform')
        
        self.pos_embed = Embedding(input_dim=num_patches, output_dim=projection_dims)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        patch_embed = self.dense(patch)

        broadcast_shape = tf.where([True, False, False], tf.shape(patch), [0, 1, self.projection_dims])
        cls_embed = tf.broadcast_to(self.class_embed, [tf.shape(patch)[0], 1, self.projection_dims])
        patch_embed = tf.concat([cls_embed, patch_embed], axis=1)


        encode = patch_embed + self.pos_embed(pos)

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
            num_heads=self.num_heads, key_dim=self.projection_dims, dropout=0.1
        )

        self.MLP_Encoder = FullyConnected(self.transformer_units, 0.1)
        self.Flatten = Flatten()
        self.Dropout = Dropout(0.5)
        #self.MLP_Head = FullyConnected([2048, 1024], 0.5)
        self.DenseClass = Dense(units=num_classes)

    def call(self, input, training = False):

        probe_list = []
        for id in range(self.num_encoders):
            x = self.Norm1(input)
            attention_out = self.AttentionHead(x, x)
            sum_1 = attention_out + input
            x = self.Norm2(x)
            x = self.MLP_Encoder(x)
            x += sum_1
            if self.insert_probes == True:
                probe = tf.stop_gradient(Probe(self.num_classes, id)(x))
                probe_list.append(probe)
            input = x

        x = self.Norm3(x)
        #x = self.GlobAvg1D(x)
        #x = self.Flatten(x)
        #x = self.Dropout(x)
        #x = self.MLP_Head(x)
        x = self.DenseClass(x[:,0])


        return [x, probe_list]


"""
References:
(1) https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py
"""
