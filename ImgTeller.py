from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
#you should find the other coe that uses weights


#refference:https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/training'
validation_data_dir = 'data/validation'
nb_train_samples = 3010
nb_validation_samples = 1318
epochs = 1
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#relu and sigmoid are both activation functions
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#relu is a linear max(0,x) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#this has two outputs which has been added to the first connected layer(the flattened one)
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#sigmoid is a function 1/(1+e)^-x
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights('first_try.h5');
			  
			  
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

img = image.load_img('test/animemedia.jpg',target_size=(150,150)); #this is a PIL Image
#issue is declaring the image before it goes in model/predict
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
prediction = model.predict(img)
print (prediction);

#img = test_datagen.flow_from_directory(
#    'test/image.jpg',
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='binary')