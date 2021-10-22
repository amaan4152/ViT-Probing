import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Probe:
	def __init__(self):
		pass
	
	def __call__(self, input):
		return layers.Softmax(input)