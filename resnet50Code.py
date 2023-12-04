import matplotlib.pyplot as plotter_lib
import matplotlib.pyplot as plt
import numpy as np
import PIL as image_lib
import tensorflow as tflow
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
import pathlib

directory = "C:\Users\jemst\Documents\Uni\Year_3\Project\project_dataset"
data_directory = pathlib.Path(directory)

img_height,img_width=180,180
batch_size=32

train_ds = tflow.keras.preprocessing.image_dataset_from_directory(data_directory, validation_split=0.2, subset="training", seed=123, label_mode='categorical', image_size=(img_height, img_width), batch_size=batch_size)
validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(data_directory, validation_split=0.2, subset="validation", seed=123, label_mode='categorical', image_size=(img_height, img_width), batch_size=batch_size)

plotter_lib.figure(figsize=(10, 10))
epochs=10
for images, labels in train_ds.take(1):
    for var in range(6):
        ax = plt.subplot(3, 3, var + 1)
        plotter_lib.imshow(images[var].numpy().astype("uint8"))
        plotter_lib.axis("off")

resnet_model = Sequential()
pretrained_model= tflow.keras.applications.ResNet50(include_top=False, input_shape=(180,180,3), pooling='avg',classes=5, weights='imagenet')

for each_layer in pretrained_model.layers:
    each_layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = resnet_model.fit(train_ds, validation_data=validation_ds, epochs=epochs)

plotter_lib.figure(figsize=(8, 8))
epochs_range= range(epochs)
plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")
plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plotter_lib.axis(ymin=0.4,ymax=1)
plotter_lib.grid()
plotter_lib.title('Model Accuracy')
plotter_lib.ylabel('Accuracy')
plotter_lib.xlabel('Epochs')
plotter_lib.legend(['train', 'validation'])
plotter_lib.show()
plotter_lib.savefig('output-plot.png') 