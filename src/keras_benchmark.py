import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


# Input_shape
train_number = 100
test_number = 32
input_shape = (256, 256, 3)

# Generate dummy data
x_train = np.random.random((train_number,)+input_shape)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(train_number, 1)), num_classes=10)
x_test = np.random.random((test_number,)+input_shape)
y_test = keras.utils.to_categorical(np.random.randint(10, size=(test_number, 1)), num_classes=10)

model = Sequential()
# input: 256x256 images with 3 channels -> (256, 256, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

epochs=100
print("Fitting a model with random data for {} epochs".format(epochs))
model.fit(x_train, y_train, batch_size=32, epochs=epochs)

#score = model.evaluate(x_test, y_test, batch_size=32)