import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from sklearn import metrics
import tensorflow as tf

#Code adapted from https://medium.com/@codeai99/image-classification-on-cnn-using-resnet50-87d5a336fe4a
NUM_CLASSES = 5
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 8
FREEZE_LAYERS = 2
NUM_EPOCHS = 10
WEIGHTS_FINAL = 'modelresnet50.h5'

#Paths to folders holding images
TRAIN_DATASET_PATH = './split_project_dataset/train'
VAL_DATASET_PATH = './split_project_dataset/validation'

#Load images into tensorflow Datasets
#ImageDataGenerator is deprecated so using image_dataset_from_directory as it says to in the tensorflow docs https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
trainDataset = image_dataset_from_directory(TRAIN_DATASET_PATH, labels='inferred', label_mode="categorical", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic")
# testDataset = image_dataset_from_directory(TEST_DATASET_PATH, labels='inferred', label_mode="categorical", shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic")
valDataset = image_dataset_from_directory(VAL_DATASET_PATH, labels='inferred', label_mode="categorical", shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic")

#Set up model
resNet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = resNet.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
model = Model(inputs=resNet.input, outputs=output_layer)
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainDataset, validation_data = valDataset, epochs = NUM_EPOCHS)

scores = model.evaluate(valDataset, verbose=1)
print('Accuracy is %s' %(scores[1]*100))

Y_pred = model.predict(valDataset)
y_pred = np.argmax(Y_pred, axis=1)
Y_true = tf.concat([y for x, y in valDataset], axis=0)
y_true = np.argmax(Y_true, axis=1)

print("Confusion matrix")
print(metrics.confusion_matrix(y_true, y_pred))

print("Classification report")
print(metrics.classification_report(y_true, y_pred))

model.save_weights(WEIGHTS_FINAL)