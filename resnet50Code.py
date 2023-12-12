import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from sklearn import metrics
import tensorflow as tf

NUM_CLASSES = 5
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 8
FREEZE_LAYERS = 2
NUM_EPOCHS = 10
WEIGHTS_FINAL = 'modelresnet50.h5'

train_dir = './split_project_dataset/train'
valid_dir = './split_project_dataset/validation'

train_batches = image_dataset_from_directory(
    directory=train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred',
    label_mode='categorical',
    class_names=['brushing_teeth', 'cutting_nails', 'doing_laundry', 'folding_clothes', 'washing_dishes'],
    shuffle=True, 
    interpolation="bicubic")

valid_batches = image_dataset_from_directory(
    directory=valid_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred',
    label_mode='categorical',
    class_names=['brushing_teeth', 'cutting_nails', 'doing_laundry', 'folding_clothes', 'washing_dishes'],
    shuffle=False, 
    interpolation="bicubic")

AUTOTUNE = tf.data.AUTOTUNE

train_batches = train_batches.cache().prefetch(buffer_size=AUTOTUNE)
valid_batches = valid_batches.cache().prefetch(buffer_size=AUTOTUNE)

classify = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = classify.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
classify_final = Model(inputs=classify.input, outputs=output_layer)
for layer in classify_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in classify_final.layers[FREEZE_LAYERS]:
    layer.trainable = True

classify_final.summary()
classify_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
classify_final.fit_generator(train_batches, steps_per_epoch = train_batches.samples // BATCH_SIZE, validation_data = valid_batches, validation_steps = valid_batches.samples // BATCH_SIZE, epochs = NUM_EPOCHS)
scores = classify_final.evaluate_generator(valid_batches, steps=1000, verbose=1)
print('Accuracy is %s' %(scores[1]*100))
Y_pred = classify_final.predict_generator(valid_batches, steps=19)
y_pred = np.argmax(Y_pred, axis=1)
print("Confusion matrix")
print(metrics.confusion_matrix(valid_batches.classes, y_pred))
print("Classification report")
print(metrics.classification_report(valid_batches.classes, y_pred))
classify_final.save_weights(WEIGHTS_FINAL)