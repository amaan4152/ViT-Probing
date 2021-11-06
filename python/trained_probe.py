import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
from test2 import BATCH_SIZE, get_EncoderOutputs

#----------Import data----------#
'''
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
'''
x_train, y_train, x_test, y_test = get_EncoderOutputs()

#----------Create Probes----------#
NUM_PROBES = 12
probe_list = []

for _ in range(NUM_PROBES):
	probe = tf.keras.Sequential(
		[
			Flatten(),
			Dense(10, activation='softmax')
		]
	)

	probe_list.append(probe)

#----------Train Probes----------#
EPOCHS = 100
probe_num = 1
for probe in probe_list:
    print(f'=== TRAINING PROBE #{probe_num}===')
    probe.compile(optimizer='adam', 
				  loss='sparse_categorical_crossentropy', 
				  metrics=['accuracy'])

    probe.fit(x_train[probe_num - 1, :], y_train, 
			  epochs=EPOCHS, 
			  batch_size=BATCH_SIZE,
			  steps_per_epoch=(0.9 * x_train.shape[1] // BATCH_SIZE),
			  validation_split=0.1)
    probe_num += 1
print('=== TRAINING COMPLETE ===')


#----------Test Probes----------#
probe_num = 1
probe_accuracies = []
for probes in probe_list:
    print('Evaluating Probe ' + str(probe_num) + '!')
    result = probe.evaluate(x_test[probe_num - 1, :], y_test)
    accuracy = np.asarray(result[1], dtype=np.float32) * 100
    probe_accuracies.append(accuracy)
    print('Evaluation of Probe ' + str(probe_num) + ' COMPLETED!')
    probe_num += 1
print('EVALUATION COMPLETED!')


#----------Print Results----------#
encoders = range(1, NUM_PROBES + 1)
plt.figure()
plt.title('Trained ViT Probe Accuracies (CIFAR10)')
plt.xlabel('Encoder #')
plt.ylabel('Accuracy [%]')
plt.plot(encoders, probe_accuracies)
plt.savefig('trained_probes.png')

		