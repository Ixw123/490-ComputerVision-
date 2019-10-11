from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.datasets import mnist
from keras.models import load_model
import keras.utils
import numpy as np
import sys
import Network1
import Network2

RAND_SEED=5280

def saveModelAndWeights(model, modelName):
	model_json = model.to_json()
	with open(modelName + ".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(modelName + ".h5")

def loadModelAndWeights(modelName):
	json_file = open(modelName + ".json", "r")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(modelName + ".h5")

	loaded_model.compile(optimizer='adagrad',
					loss='categorical_crossentropy',
					metrics=['accuracy'])

	return loaded_model

def saveModel(model, modelName):
	model.save(modelName + "_ALL.h5")

def loadModel(modelName):
	return load_model(modelName + "_ALL.h5")

def parseArgs():
	if len(sys.argv) < 2:
		print("ERROR: Must pass in either '1' or '2'!")
		exit()

	if sys.argv[1] == '1':
		Network = Network1
		NetworkName = "Network1"
	elif sys.argv[1] == '2':
		Network = Network2
		NetworkName = "Network2"
	else:
		print("ERROR: Invalid network ID specified!")
		exit()

	return Network, NetworkName

def createDataGenerator(x_train):
	datagen = ImageDataGenerator(
		samplewise_center=True, 
		samplewise_std_normalization=True, 
		rotation_range=20, 
		width_shift_range=0.2, 
		height_shift_range=0.2, 
		horizontal_flip=True)
	#datagen.fit(x_train)
	return datagen

def preprocessImages(images, x_train):
	datagen = createDataGenerator(x_train)
	return datagen.standardize(images)

def prepareMNIST():
	classCnt = 10

	# Load MNIST data
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Reshape image data so channel is last value
	x_train = np.reshape(x_train, list(x_train.shape) + [1])
	x_test = np.reshape(x_test, list(x_test.shape) + [1])
	
	# Convert to one-hot vectors	
	one_hot_train = keras.utils.to_categorical(y_train, num_classes=classCnt)
	one_hot_test = keras.utils.to_categorical(y_test, num_classes=classCnt)
	one_hot_train = one_hot_train.astype('float32')
	one_hot_test = one_hot_test.astype('float32')

	# Normalize
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	
	return x_train, one_hot_train, x_test, one_hot_test, classCnt


