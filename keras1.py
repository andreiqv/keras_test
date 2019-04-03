from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

model = Sequential()
input_shape=(320,320,3) #this is the input shape of an image 320x320x3
model.add(Conv2D(48, (3, 3), activation='relu', input_shape= input_shape))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))