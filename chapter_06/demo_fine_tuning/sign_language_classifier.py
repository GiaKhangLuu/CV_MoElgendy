# Import lib
from keras.preprocessing.image import  ImageDataGenerator
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications import imagenet_utils
from keras.optimizers import Adam, SGD
import keras.models
from keras.metrics import categorical_crossentropy
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# Load data
train_path = 'dataset/train'
valid_path = 'dataset/valid'
test_path = 'dataset/test'

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

# Load pretrained model
# Use average pooling instead of flatten
#base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='avg')
#base_model.save('pretrained_vgg16.h5')
#base_model = keras.models.load_model('pretrained_vgg16.h5')

# Freeze first 5 layers
"""
for layer in base_model.layers[:-5]:
    layer.trainable = False
#base_model.summary()

# Create new model
last_output = base_model.output
x = Dense(10, activation='softmax', name='softmax')(last_output)
new_model = Model(inputs=base_model.input, outputs=x)
#new_model.summary()

# Compile model
from keras.callbacks import ModelCheckpoint
new_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='sign_language_model.hdf5', save_best_only=True)
history = new_model.fit_generator(train_batches, validation_data=valid_batches,
                                  steps_per_epoch=18, validation_steps=3,
                                  epochs=20, verbose=1, callbacks=[checkpointer])
"""
new_model = keras.models.load_model('sign_language_model.hdf5')

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

def load_dataset(path):
    data = load_files(path)
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

test_files, test_targets = load_dataset('dataset/test')
test_tensors = preprocess_input(paths_to_tensor(test_files))

print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'
      .format(*new_model.evaluate(test_tensors, test_targets)))

# Confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

cm_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Create confusion matrix
cm = confusion_matrix(np.argmax(test_targets, axis=1),
                      np.argmax(new_model.predict(test_tensors), axis=1))
# Plot confusion matrix
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
indexes = np.arange(len(cm_labels))
for i in indexes:
    for j in indexes:
        plt.text(j, i, cm[i, j])
plt.xticks(indexes, cm_labels)
plt.xlabel('Predicted label')
plt.yticks(indexes, cm_labels)
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()