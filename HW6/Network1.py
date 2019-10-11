# Written by Micah Church for CS490 Computer Vision
# Directly copied from Example Network Slides (Dr. Michael Reale: Computer Vision 2018) 
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

def buildModel(inputShape, classCnt):
	input_shape=inputShape
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(classCnt, activation='softmax'))

	model.compile(optimizer='adagrad',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	return model
