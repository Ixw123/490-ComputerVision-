from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import Network1
import Network2
import sys
from Common import *

def main():

	_, NetworkName = parseArgs()

	np.random.seed(RAND_SEED)

	print("Testing", NetworkName)

	# Load model	
	model = loadModel(NetworkName)

	# Prepare MNIST
	x_train, one_hot_train, x_test, one_hot_test, _ = prepareMNIST()

	# Create data generator
	datagen = createDataGenerator(x_train)

	# Print model summary
	model.summary()

	batchSize = 32

	# Evaluate model	
	norm_x_train = preprocessImages(x_train, x_train)
	scores = model.evaluate(norm_x_train, one_hot_train, batch_size=batchSize)
	print("TRAINING SCORES:", scores)

	norm_x_test = preprocessImages(x_test, x_train)
	scores = model.evaluate(norm_x_test, one_hot_test, batch_size=batchSize)
	print("TEST SCORES:", scores)

if __name__ == "__main__": main()

