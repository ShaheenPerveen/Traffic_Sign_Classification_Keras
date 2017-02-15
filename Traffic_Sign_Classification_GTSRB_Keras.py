

import numpy as np
from sklearn.cross_validation import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import cv2



nb_classes = 43
img_size = 32

## loading training data
import pickle
training_set = pickle.load(open("/home/aiml_test_user/training_set.pkl", "rb"))

## transorforming tha data
# resizing all images to one
resized_img=[]
class_labels = []
for i in range(0, len(training_set)):
    sample = training_set[i]
    image = cv2.resize(sample['img'],(img_size, img_size))
    resized_img.append(image)
    class_labels.append(sample['class'])
    
resize_imgs = np.array(resized_img)

Y = np.array(class_labels)

## dividing by 255
X = np.array(resized_img, dtype='float32')
X /= 255

## swapping axes to input into model
X = X.swapaxes(1,3)
X.shape


## splitting the data into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



## model

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, img_size, img_size), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

model = cnn_model()


lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))



## fitting the model
batch_size = 30
nb_epoch = 10

model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, Y_test))


# In[9]:

validation = model.evaluate(x_test, Y_test, verbose=1)
print('Test accuracy:', validation[1])


# In[10]:

model.summary()


# In[4]:

## testing
import pickle
testing_set = pickle.load(open("/home/aiml_test_user/testing_set.pkl", "rb"))

## preprocessing
resized_img_test=[]
for i in range(0, len(testing_set)):
    sample = testing_set[i]
    image = cv2.resize(sample['img'],(img_size, img_size))
    resized_img_test.append(image)
resize_imgs_test = np.array(resized_img_test)

X_test = np.array(resize_imgs_test, dtype='float32')
X_test /= 255

X_test= X_test.swapaxes(1,3)
X_test.shape


# In[12]:

## predicting the test data
y_pred = model.predict_classes(X_test)


# In[15]:

## saving the predicted data
np.savetxt("submission.csv", y_pred, delimiter=",")


# In[5]:

x_train.shape


# In[5]:

## data augmentation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)

datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)


# In[7]:

# Reinstallise models 

model = cnn_model()
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


# In[10]:

nb_epoch = 20
batch_size = 40
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val))


# In[11]:

## predicting the test data
y_pred = model.predict_classes(X_test)

## saving the predicted data
np.savetxt("submission.csv", y_pred, delimiter=",")


# In[13]:

## data augmentation - part 2
X_train, X_val, Y_train,Y_val = train_test_split(X, Y, test_size=0.2)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)

datagen1 = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)


# compute quantities required for featurewise normalization
datagen1.fit(X_train)

# Reinstallise models 

model = cnn_model()
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


nb_epoch = 20
batch_size = 40
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val))


# In[ ]:

## predicting the test data
y_pred = model.predict_classes(X_test)

## saving the predicted data
np.savetxt("submission.csv", y_pred, delimiter=",")

