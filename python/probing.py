import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
x_train = 
y_train =

x_test = 
y_test = 

#----------Create Probes----------#
NUM_PROBES = 12
probe_list = []

for _ in range(NUM_PROBES):
	probe = tf.keras.Sequential(
		[
			tf.keras.layers.Flatten(),
			tf.keras.Dense(10, activation='softmax')
		]
	)

	probe_list.append(probe)

#----------Train Probes----------#
EPOCHS = 100
probe_num = 1
for probes in probe_list:
	print('Training Probe ' + str(probe_num) + '!')
	probe.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	probe.fit(x_train[probe_num - 1, :], y_train, epochs=EPOCHS)
	print('Training of Probe ' + str(probe_num) + ' COMPLETED!')
print('TRAINING COMPLETED!')

#----------Test Probes----------#
probe_num = 1
probe_accuracies = []
for probes in probe_list:
	print('Evaluating Probe ' + str(probe_num) + '!')
	result = probe.evaluate(x_test, y_test)
	accuracy = np.asarray(result[1], dtype=np.float32) * 100
	probe_accuracies.append(accuracy)
	print('Evaluation of Probe ' + str(probe_num) + ' COMPLETED!')
print('EVALUATION COMPLETED!')

#----------Print Results----------#
encoders = range(1, NUM_PROBES + 1)
plt.figure()
plt.title('Trained ViT Probe Accuracies (CIFAR10)')
plt.xlabel('Encoder #')
plt.ylabel('Accuracy [%]')
plt.plot(encoders, probe_accuracies)
plt.show()

		
