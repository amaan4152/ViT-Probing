from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Dense,
    LayerNormalization,
    Conv2D,
    Reshape,
    Flatten
)

from vit_tf import (
    DataAugmentation,
    Patches
)

class Preprocessor(layers.Layer):
    def __init__(self, patch_size):
        super(Preprocessor, self).__init__()
        self.Patches = Patches(patch_size)

    def call(self, x):
        x = self.Patches(x)
        return x


class PatchConv(Model):
    def __init__(
        self,
        x_train,
        image_size,
        patch_size,
        num_classes,
        test_layer=None,
    ):
        super(PatchConv, self).__init__()
        self.DataAugmentation = DataAugmentation(image_size)
        self.DataAugmentation.layers[0].adapt(x_train)
        self.test_layer = test_layer
        self.Preprocessor = Preprocessor(
            patch_size=patch_size
        )
        if self.test_layer:
            self.Resize = Reshape((4, 15, 15, 16))
        else:
            self.Resize = Reshape((4, 16, 16, 3))

        self.Conv = Conv2D(filters=16,
            kernel_size=int(3),
            activation="relu",
            padding="VALID",
        )
        self.Flat = Flatten()
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
        x = self.Resize(x)
        x = self.Conv(x)
        outputs.append(x)
        self.outputs = outputs
        x = self.Norm3(x)
        x = self.Flat(x)
        x = self.Head(x)
        return x
    
    '''
        https://stackoverflow.com/questions/65365745/model-summary-output-is-not-consistent-with-model-definition
    '''
    def summary_model(self):
        inputs = Input(shape=(32, 32, 3))
        outputs = self.call(inputs)
        Model(inputs=inputs, outputs=outputs, name="pconv").summary()
