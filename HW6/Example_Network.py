from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(classCnt, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

labels = 10

model.fit(data, labels, epochs=10, batch_size=32)
