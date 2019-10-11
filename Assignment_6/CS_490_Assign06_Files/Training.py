from Common import *

def main():

	Network, NetworkName = parseArgs()

	print("Training", NetworkName)

	np.random.seed(RAND_SEED)
	
	batchSize = 32
	epochs = 10

	# Prepare MNIST
	x_train, one_hot_train, x_test, one_hot_test, classCnt = prepareMNIST()

	# Create data generator
	datagen = createDataGenerator(x_train)

	# Create model
	model = Network.buildModel(inputShape=(28,28,1), classCnt=classCnt)

	# Print model summary
	model.summary()

	# Train model
	model.fit_generator(datagen.flow(x_train, one_hot_train, batch_size=batchSize), 
		steps_per_epoch=len(x_train)/batchSize,		
		epochs=epochs)

	# Evaluate model
	norm_x_train = preprocessImages(x_train, x_train)
	scores = model.evaluate(norm_x_train, one_hot_train, batch_size=batchSize)
	print("TRAINING SCORES:", scores)

	norm_x_test = preprocessImages(x_test, x_train)
	scores = model.evaluate(norm_x_test, one_hot_test, batch_size=batchSize)
	print("TEST SCORES:", scores)

	# Save architecture and weights	
	saveModel(model, NetworkName)
	

if __name__ == "__main__": main()

