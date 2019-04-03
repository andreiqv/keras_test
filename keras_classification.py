from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

model = Sequential()
input_shape = (320,320,3) #this is the input shape of an image 320x320x3
model.add(Conv2D(48, (3, 3), activation='relu', input_shape= input_shape))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))




sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(X_train, X_train, batch_size=32, epochs=10, 
	validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, batch_size=32)
