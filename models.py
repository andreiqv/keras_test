import keras
layers = keras.layers

from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.mobilenet_v2 import MobileNetV2  #224x224.
#from keras.applications.mobilenet import MobileNet  #224x224.

OUTPUT_NAME = 'output'


def model_ResNet50(inputs):

	base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', 
		input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_ResNet50')	
	return model

# --------------------

def conv(x, f, k, s=1, p='SAME', a='relu'):
	x = layers.Conv2D(
		filters=f,
		kernel_size=(k, k),
		strides=(s, s),
		padding=p,
		activation=a, # relu, selu
		#kernel_regularizer=regularizers.l2(0.01),
		use_bias=True)(x)
	return x

maxpool = lambda x, p=2, s=1: layers.MaxPool2D(pool_size=p, strides=s)(x)	
maxpool2 = lambda x, p=2: layers.MaxPool2D(pool_size=p)(x)	
bn = lambda x: layers.BatchNormalization()(x)

def model_3(inputs):
	""" model_first_3 == model_first2_1
	--
	77: val_miou: 0.8053
	"""
	x = inputs
	x = conv(x, f=8, k=3, s=1, p='VALID')
	x = maxpool(x)  # 64
	
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x)
	
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x)
	x = bn(x)
	x = conv(x, f=32, k=3, s=2, p='VALID')
	x = maxpool(x)
	
	x = bn(x)
	x = conv(x, f=32, k=3, s=1, p='VALID')
	x = maxpool(x)
	
	x = bn(x)	
	x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1000, activation='sigmoid')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	#x = layers.Dense(5, activation=None)(x)
	model = keras.Model(inputs, x, name='model_first_3')
	return model



def cnn_128(inputs, num_classes):
	""" Epoch 44/500 - 456s 381ms/step 
	- loss: 1.4250e-04 - accuracy: 0.9999 - miou: 0.9370 - 
	val_loss: 0.0053 - val_accuracy: 0.9999 - val_miou: 0.7339
	"""
	x = inputs 
	x = conv(x, 8, 5)
	#x = conv(x, 8, 5)
	x = maxpool(x)  # 64
	x = conv(x, 16, 3)
	#x = conv(x, 16, 3)
	x = maxpool(x)  # 32
	x = conv(x, 16, 3)
	#x = conv(x, 16, 3)
	x = maxpool(x)  # 16
	x = conv(x, 32, 3)
	#x = conv(x, 32, 3)
	x = maxpool(x)  # 8

	x = layers.Flatten()(x)
	#x = layers.Dropout(0.5)(x)
	#x = layers.Dense(1000, activation='elu')(x)
	#x = layers.Dropout(0.5)(x)
	x = layers.Dense(num_classes, activation='softmax', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='cnn_128')
	
	return model