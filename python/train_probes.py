import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
import numpy as np

BATCH_SIZE = 1024

# ----------Import data----------#
"""
Import such that:
x_train: Matrix of size (NUM_PROBES, NUM_IMAGES) 
	containing matrices of outputs of encoder
y_train: Vector of size (NUM_IMAGES, ) 
	containing vectors of the true labels
	for each image
x_test: Matrix of size (NUM_PROBES, NUM_IMAGES)
	containing matrices of outputs of encoder
y_test: Vector of size (NUM_IMAGES, )
	containing vectors of true labels

Make sure that training >> testing
"""


def train_probes(data):
    x_train, y_train, x_test, y_test = data

    # ----------Create Probes----------#
    NUM_PROBES = len(x_train)
    print(f"NUM PROBES: {NUM_PROBES}")
    probe_list = []

    for _ in range(NUM_PROBES):
        # Try changing this (don't use sequential)
        # flatten: untrustworthy!
        probe = tf.keras.Sequential(
            [Dense(10, activation="softmax")]
        )
        probe_list.append(probe)

    # ----------Train Probes----------#
    EPOCHS = 100
    probe_num = 1
    call_ES = EarlyStopping(patience=5)
    for probe in probe_list:

        print(f"=== TRAINING PROBE #{probe_num}===")
        probe.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        probe.fit(
            tf.reshape(x_train[probe_num - 1], (50000, -1)),
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            steps_per_epoch=(0.8 * x_train[0].shape[0] // BATCH_SIZE),
            validation_split=0.2,
            callbacks=[call_ES],
        )
        probe_num += 1
    print("=== TRAINING COMPLETE ===")

    # ----------Test Probes----------#
    probe_num = 1
    probe_accuracies = []
    for probe in probe_list:
        print("Evaluating Probe " + str(probe_num) + "!")
        result = probe.evaluate(tf.reshape(x_test[probe_num - 1], (10000, -1)), y_test)
        accuracy = np.asarray(result[1], dtype=np.float32) * 100
        probe_accuracies.append(accuracy)
        print("Evaluation of Probe " + str(probe_num) + " COMPLETED!")
        probe_num += 1
    print("EVALUATION COMPLETED!")

    # ----------Print Results----------#
    x_vals = np.arange(1, NUM_PROBES + 1)
    if NUM_PROBES == 3:
        x_vals = np.arange(1, NUM_PROBES + 2)
        x_vals = x_vals[x_vals != 2]
    elif NUM_PROBES == 14: 
        x_vals = np.arange(1, NUM_PROBES + 2)
        x_vals = x_vals[x_vals != 2]

    return {"x": x_vals, "y": probe_accuracies}
