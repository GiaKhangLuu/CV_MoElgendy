from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications import vgg16
from keras.applications import mobilenet
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import cv2

# Load data
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         target_size=(224, 224),
                                                         batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                         target_size=(224, 224),
                                                         batch_size=30)
test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                        target_size=(224, 224),
                                                        batch_size=50,
                                                        shuffle=False)

# Load pretrained weights from vgg16
base_model = vgg16.VGG16(weights="imagenet", include_top="False", input_shape=(224, 224, 3))
base_model.save('vgg16_as_feature_extractor.h5')

base_model = keras.models.load_model('vgg16_as_feature_extractor.h5')

# Freeze layer
for layer in base_model.layers:
    layer.trainable = False

# Add new classifier
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

x = Flatten()(last_output)

x = Dense(64, activation='relu', name='FC_2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='softmax')(x)

new_model = Model(inputs=base_model.input, outputs=x)
#new_model.summary()

# Compile model
new_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
new_model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches,
                        validation_steps=2, epochs=20, verbose=2)
new_model.save('new_model.h5')

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

def load_dataset(path):
    data = load_files('data/test')
    paths = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']))
    return paths, targets

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) # Cvt PIL.Image.Image to a 3D tensor w/ shape (224, 224, 3)
    return np.expand_dims(x, axis=0) # Cvt 3D tensor to 4D tensor w/ shape (1, 224, 224, 3)

def paths_to_tensor(img_paths):
    list_of_tensor = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensor)

test_files, test_targets = load_dataset('data/test')

test_tensors = preprocess_input(paths_to_tensor(test_files))

# Evaluate performance
print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'
      .format(*new_model.evaluate(test_tensors, test_targets)))
