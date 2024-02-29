import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Rescaling, RandomFlip, RandomRotation
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import tensorflow as tf
from keras import Sequential

#Code modified from https://medium.com/@codeai99/image-classification-on-cnn-using-resnet50-87d5a336fe4a
NUM_CLASSES = 5
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 8
FREEZE_LAYERS = 2
NUM_EPOCHS = 10

#Path to folder holding images
DATASET_PATH = "./project_dataset"

classes = ["brushing_teeth", "cutting_nails", "doing_laundry", "folding clothes", "washing dishes"]
numClasses = [0, 1, 2, 3, 4]

#Load images into tensorflow Datasets
#ImageDataGenerator (from copied code) is deprecated so using image_dataset_from_directory as it says to in the tensorflow docs https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
trainDataset = image_dataset_from_directory(DATASET_PATH, labels="inferred", label_mode="categorical", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic", validation_split=0.2, subset="training", seed=42)

valDataset = image_dataset_from_directory(DATASET_PATH, labels="inferred", label_mode="categorical", shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic", validation_split=0.2, subset="validation", seed=42)

class_names = trainDataset.class_names
print(class_names)

#Augmenting data
data_augmentation = Sequential([RandomFlip("horizontal"), RandomRotation(0.1), Rescaling(1./255)])

AUTOTUNE = tf.data.AUTOTUNE

trainDataset.map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=AUTOTUNE)

trainDataset = trainDataset.prefetch(buffer_size=AUTOTUNE)
valDataset = valDataset.prefetch(buffer_size=AUTOTUNE)

#Set up model
resNet = ResNet50(include_top=False, weights="imagenet", input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = resNet.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation="softmax", name="softmax")(x)
model = Model(inputs=resNet.input, outputs=output_layer)
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# model.summary()

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(trainDataset, validation_data=valDataset, epochs=NUM_EPOCHS)

#Print out metrics
scores = model.evaluate(valDataset, verbose=1)
print("Accuracy is %s" %(scores[1]*100))

Y_pred = model.predict(valDataset)
y_pred = np.argmax(Y_pred, axis=1)
Y_true = tf.concat([y for x, y in valDataset], axis=0)
y_true = np.argmax(Y_true, axis=1)

print(y_true)

print("Confusion matrix")
cm = confusion_matrix(y_true, y_pred, labels=numClasses)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=numClasses)

disp.plot()
plt.show()

print("Classification report")
print(classification_report(y_true, y_pred))

model.save("model.h5")