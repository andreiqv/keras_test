import keras
layers = keras.layers

from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.mobilenet_v2 import MobileNetV2  #224x224.
#from keras.applications.mobilenet import MobileNet  #224x224.

OUTPUT_NAME = 'output'



def cnn_128(inputs, num_classes):
		
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, INPUT_SIZE, INPUT_SIZE)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# the model so far outputs 3D feature maps (height, width, features)
	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(8))
	model.add(Activation('softmax'))
	return model