from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten


class Probe(layers.Layer):
    def __init__(self, num_classes, id):
        super(Probe, self).__init__()
        self.Flat = Flatten()
        self.Probe = Dense(
            units=num_classes, activation="softmax", name="Probe_" + str(id)
        )

    def call(self, x):
        x = self.Flat(x)
        x = self.Probe(x)
        return x
