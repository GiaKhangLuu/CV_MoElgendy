from keras.preprocessing.image import  load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# Download the pretrained vgg16, include_top=True bc we want to use vgg16 as classifier
model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

# Load and preprocess input
img = load_img('dog.jpeg', target_size=(224, 224))
img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
img = preprocess_input(img)

# Predict image
y_hat = model.predict(img)
label = decode_predictions(y_hat)
label = label[0][0]
print('%s: %.2f%%' % (label[1], label[2]*100))