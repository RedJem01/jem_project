from matplotlib import pyplot
from keras.applications.resnet50 import ResNet50
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Rescaling, RandomFlip, RandomRotation
import tensorflow as tf
from keras import Sequential

#Code modified from https://medium.com/@codeai99/image-classification-on-cnn-using-resnet50-87d5a336fe4a
NUM_CLASSES = 5
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 8
FREEZE_LAYERS = 2
NUM_EPOCHS = 10

#Paths to folders holding images
TRAIN_DATASET_PATH = "./split_project_dataset/train"
VAL_DATASET_PATH = "./split_project_dataset/validation"

classes = ["brushing_teeth", "cutting_nails", "doing_laundry", "folding clothes", "washing dishes"]

#Load images into tensorflow Datasets
#ImageDataGenerator is deprecated so using image_dataset_from_directory as it says to in the tensorflow docs https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
trainDataset = image_dataset_from_directory(TRAIN_DATASET_PATH, labels="inferred", label_mode="categorical", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic")

# testDataset = image_dataset_from_directory(TEST_DATASET_PATH, labels='inferred', label_mode="categorical", shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic")
valDataset = image_dataset_from_directory(VAL_DATASET_PATH, labels="inferred", label_mode="categorical", shuffle=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, interpolation="bicubic")

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

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(trainDataset, validation_data = valDataset, epochs = NUM_EPOCHS)

print(history.history.keys())

# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

model.save("model.h5")