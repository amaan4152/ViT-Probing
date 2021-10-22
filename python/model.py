import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from probe import Probe

'''
ViT Architecture
'''
class VisualTransformer:
	'''
	Transformer encoder class
	'''
	class Transformer:
		def __init__(self, num_heads, projection_dim):
			self.num_heads = num_heads
			self.projection_dim = projection_dim
			self.transformer_units = [
    			2*projection_dim,
    			projection_dim,
			]

		def __call__(self, x):
			'''Layer normalization 1'''
			x1 = layers.LayerNormalization(epsilon=1e-6)(x)

			'''Create a multi-head attention layer'''
			attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1, x1)
			
			'''Skip connection 1'''
			x2 = layers.Add()([attention_output, x])

			'''Layer normalization 2'''
			x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

			'''Multi-layer perceptron'''
			for units in self.transformer_units:
				x3 = layers.Dense(units, activation=tf.nn.gelu)(x3)
				x3 = layers.Dropout(0.1)(x3)

			'''Skip connection 2'''
			x = layers.Add()([x3, x2])

			return x

	def __init__(self):
		num_classes = 100
		num_transformer_layers = 8
		projection_dim = 64
		num_heads = 4
		input_shape = (32, 32, 3)
		image_size = 72
		patch_size = 6
		num_patches = (image_size // patch_size) ** 2
		mlp_head_units = [2048, 1024]

		self.inputs = layers.Input(shape=input_shape)
		augmented_data = self.augment_data(image_size)(self.inputs)
		patches = self.create_patches(patch_size, augmented_data)
		encoded_patches = self.encode_patches(num_patches, projection_dim, patches)
		x = encoded_patches

		for _ in range(num_transformer_layers):
			x = self.Transformer(num_heads, projection_dim)(x)

		representation = layers.LayerNormalization(epsilon=1e-6)(x)
		representation = layers.Flatten()(representation)
		representation = layers.Dropout(0.5)(representation)

		'''Multi-layer perceptron'''
		for units in mlp_head_units:
			x = layers.Dense(units, activation=tf.nn.gelu)(representation)
			x = layers.Dropout(0.5)(x)
		features = x

		self.logits = layers.Dense(num_classes)(features)

		self.model = keras.Model(inputs=self.inputs, outputs=self.logits)
		self.model.summary()
	
	def __call__(self):
		return self.model	

	def augment_data(self, image_size):
		data_augmentation = keras.Sequential([
        	layers.Normalization(),
        	layers.Resizing(image_size, image_size),
        	layers.RandomFlip("horizontal"),
        	layers.RandomRotation(factor=0.02),
        	layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    	], 
		name="data_augmentation"
		)
		return data_augmentation

	def create_patches(self, patch_size, images):
		batch_size = tf.shape(images)[0]
		patches = tf.image.extract_patches(
			images=images,
			sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
		patch_dims = patches.shape[-1]
		patches = tf.reshape(patches, [batch_size, -1, patch_dims])
		return patches
	
	def encode_patches(self, num_patches, projection_dim, patches):
		projection = layers.Dense(units=projection_dim)
		position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
		positions = tf.range(start=0, limit=num_patches, delta=1)
		encoded_patches = projection(patches) + position_embedding(positions)
		return encoded_patches

	def get_outputs(self):
		outputs = []
		for layer in self.model.layers:
			if (layer.name[:3] == 'add') and (int(layer.name[4:]) % 2 != 0):
				outputs.append(layer.output)
		return outputs

	def probe_layers(self, num_classes, outputs):
		probe_outputs = []
		for output in outputs:
			probe_outputs.append(Probe(num_classes)(output))
		
			

		
VisualTransformer()()

'''
References:
(1) https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py
'''