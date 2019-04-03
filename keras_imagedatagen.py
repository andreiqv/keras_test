# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

BATCH_SIZE = 8
INPUT_SIZE = 128

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)

"""
train_datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
"""

def check_datagen_on_image(image_path):
	print(image_path)
	img = load_img(image_path)  # this is a PIL image
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory	
	#for batch in datagen.flow(x, batch_size=1,
	#				save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
	
	iterator = train_datagen.flow(x, batch_size=1, save_to_dir='preview', 
							save_prefix='cat', save_format='jpeg')
	for _ in range(2):
		batch = iterator.next()

check_datagen_on_image('/mnt/lin2/datasets/natural_images/cat/cat_0008.jpg')


# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/mnt/lin2/datasets/natural_images',  # this is the target directory
        target_size=(INPUT_SIZE, INPUT_SIZE),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical')  

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        '/mnt/lin2/datasets/natural_images',
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical') #'binary' - # since we use binary_crossentropy loss, we need binary labels

i = 0
for batch in train_generator:
	print(batch[0].shape)
	print(batch[1].shape)
	i += 1
	if i > 2: break

print(dir(train_generator))
print(train_generator.num_classes)
print(train_generator.classes)
print(len(train_generator.filenames))
print(train_generator.total_batches_seen)

num_classes = train_generator.num_classes
#-----------------

from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense

from keras.applications.resnet50 import ResNet50

def model_ResNet50(inputs):
	base_model = ResNet50(weights='imagenet', include_top=False, 
		pooling='avg', input_tensor=inputs)
	x = base_model.output
	x = Dense(8, activation='softmax')(x)
	model = Model(inputs=inputs, outputs=x, name='keras_ResNet50')	
	return model

inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input')	
#model = model_ResNet50(inputs) 

from models import cnn_128
model = cnn_128(inputs, num_classes=num_classes)

"""
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
"""

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='binary_crossentropy', 
#			optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', 
			optimizer='rmsprop', metrics=['accuracy'])


model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // train_generator.batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // validation_generator.batch_size)
#model.save_weights('first_try.h5')  # always save your weights after training or during training

#model.fit(X_train, X_train, BATCH_SIZE=32, epochs=10, 
#	validation_data=(x_val, y_val))
#score = model.evaluate(x_test, y_test, BATCH_SIZE=32)

