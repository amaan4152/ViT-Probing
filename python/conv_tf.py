import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
)

from vit_tf import (
	DataAugmentation,
	Patches,
	PatchEncoder,
	Preprocessor,
	FullyConnected, 
)

class PatchConv(Model):
    def __init__(
        self,
        x_train,
        image_size,
        num_patches,
        patch_size,
        num_encoders,
        num_heads,
        num_classes,
        projection_dims,
        test_layer=None,
    ):
        super(PatchConv, self).__init__()
        self.DataAugmentation = DataAugmentation(image_size)
        self.DataAugmentation.layers[0].adapt(x_train)
        self.test_layer = test_layer
        self.Preprocessor = Preprocessor(
            num_patches=num_patches,
            patch_size=patch_size,
            projection_dims=projection_dims,
        )
        #self.Conv = Conv2D()
        self.Norm3 = LayerNormalization(epsilon=1e-6)
        self.Head = Dense(units=num_classes)

    def call(self, x):
        layer_features = []
        outputs = []
        layer_features.append(x)
        x = self.DataAugmentation(x)
        if self.test_layer:
            x = self.test_layer(x)
            layer_features.append(x)
            # layer_features = np.array(layer_features)
        x = self.Preprocessor(x)
        layer_features.append(x)
        outputs.extend(layer_features)
        #outputs.extend(self.Encoder.encoder_features)
        self.outputs = outputs
        x = self.Norm3(x)
        x = self.Head(x[:, 0])
        return x