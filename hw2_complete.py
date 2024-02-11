


### Add lines to import modules as needed
import numpy as np
from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import layers, models

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Add
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Add, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


def calculate_params_and_mac(input_size, filters, kernel_size, stride, padding):
    output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
    params = (kernel_size**2) * input_size * filters + filters  # Weights + biases
    mac_ops = input_size * output_size * output_size * filters * (kernel_size**2)
    return params, mac_ops, output_size


# In[3]:


def build_model1():
    model = Sequential() # Add code to define model 1.

    # Conv 2D: 32 filters, 3x3 kernel, stride=2, "same" padding
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    # Conv 2D: 64 filters, 3x3 kernel, stride=2, "same" padding
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Conv 2D: 128 filters, 3x3 kernel, stride=2, "same" padding
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 
    # Four more pairs of Conv2D+Batchnorm, with default stride=1
    for _ in range(4):
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))


    # MaxPooling, 4x4 pooling size, 4x4 stride
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    # Flatten
    model.add(Flatten())

    # Dense (Fully Connected), 128 units
    model.add(Dense(128))
    model.add(BatchNormalization())

    # Dense (Fully Connected), 10 units
    model.add(Dense(10))

    return model


# In[4]:


def build_model2():
    model = Sequential() # Add code to define model .

    # Conv 2D: 32 filters, 3x3 kernel, stride=2, "same" padding
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    # Conv 2D: 64 filters, 3x3 kernel, stride=2, "same" padding
    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())

    # Conv 2D: 128 filters, 3x3 kernel, stride=2, "same" padding
    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(Conv2D(128, (1, 1), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())

    # Four more pairs of Conv2D+Batchnorm, with default stride=1
    for _ in range(4):
        model.add(DepthwiseConv2D((3, 3), padding='same', use_bias=False))
        model.add(Conv2D(128, (1, 1), strides=(1, 1), padding='valid'))
        model.add(BatchNormalization())

    # MaxPooling, 4x4 pooling size, 4x4 stride
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    # Flatten
    model.add(Flatten())

    # Dense (Fully Connected), 128 units
    model.add(Dense(128))
    model.add(BatchNormalization())

    # Dense (Fully Connected), 10 units
    model.add(Dense(10))

    return model


# In[5]:





# In[6]:


def build_model3():
    inputs = layers.Input(shape=(32, 32, 3))

    y = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    y = layers.Conv2D(128, (1, 1), strides=(5, 5), padding='same', activation='relu')(y)
    y = layers.Add()([x, y])
    y = layers.Dropout(0.5)(y)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(y)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    y = layers.Add()([x, y])
    y = layers.Dropout(0.5)(y)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(y)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    y = layers.Add()([x, y])
    y = layers.Dropout(0.5)(y)

    x = layers.MaxPooling2D(pool_size=(4, 4))(y)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(10)(x)

    model = Model(inputs, outputs)
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # model.summary()
    return model

# In[29]:





def build_model50k():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(32, kernel_size=(3, 3), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(SeparableConv2D(64, kernel_size=(3, 3), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model





# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Split the training set into training and validation subsets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    # Display the shapes of the sets
    print("Training set shape:", train_images.shape)
    print("Validation set shape:", val_images.shape)
    print("Test set shape:", test_images.shape)

  ########################################
  ## Build and train model 1
    model1 = build_model1()
    model1.summary()
    
   
    
  # compile and train model 1.
    model1.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    # Train the model for 50 epochs
    history = model1.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))
    
    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model 1 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

      # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model 1 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
        
    # Evaluate the model on the test set
    test_loss, test_acc = model1.evaluate(test_images, test_labels)
    print("\nTest accuracy:", test_acc)
    
  ## Build, compile, and train model 2 (DS Convolutions)
#     model2 = build_model2()
 ########################################
  ## Build and train model 2
    model2 = build_model2()
    model2.summary()
  
    model2.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model2.fit(train_images, train_labels, epochs=1, batch_size=128, validation_data=(val_images, val_labels))

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model 2 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

      # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model 2 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

      # Evaluate the model on the test set
    test_loss, test_acc = model2.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)

  ### Repeat for model 3 and your best sub-50k params model
    model3 = build_model3()
    model3.summary()
    model3.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Train the model for 50 epochs
    history = model3.fit(train_images, train_labels, epochs=1, batch_size=128, validation_data=(val_images, val_labels))

    
    
# Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model 3 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

  # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model 3 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

  # Evaluate the model on the test set
    test_loss, test_acc = model3.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)
    
    model50k = build_model50k()
    model50k.summary()
    model50k.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Train the model for 50 epochs
    history = model50k.fit(train_images, train_labels, epochs=1, batch_size=128, validation_data=(val_images, val_labels))



    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model 50k accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model 50k loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_acc = model50k.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy:', test_acc)



model1 = build_model1()

model1.save('build_model1.h5')


# Load the image and resize it to 32x32 pixels
img = Image.open("/test_image_airplane.ext")
img = img.resize((32, 32))

  # Convert the image to a numpy array
x = np.array(img)
x = x.reshape((1,) + x.shape)  # Add batch dimension

  # Load the trained model
model = keras.models.load_model("build_model1.h5")

  # Make a prediction on the image
pred = model.predict(x)

  # Get the predicted class label
class_idx = np.argmax(pred)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_name = class_names[class_idx]

print("The predicted class label is:", class_name)
model2 = build_model2()

model2.save('build_model2.h5')

img = Image.open("/test_image_airplane.ext")
img = img.resize((32, 32))

  # Convert the image to a numpy array
x = np.array(img)
x = x.reshape((1,) + x.shape)  # Add batch dimension

  # Load the trained model
model = keras.models.load_model("build_model2.h5")

  # Make a prediction on the image
pred = model.predict(x)

  # Get the predicted class label
class_idx = np.argmax(pred)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_name = class_names[class_idx]


print("The predicted class label is:", class_name)

model3 = build_model3()

model3.save('build_model3.h5')

img = Image.open("/test_image_airplane.ext")
img = img.resize((32, 32))

  # Convert the image to a numpy array
x = np.array(img)
x = x.reshape((1,) + x.shape)  # Add batch dimension

  # Load the trained model
model = keras.models.load_model("build_model3.h5")

  # Make a prediction on the image
pred = model.predict(x)

  # Get the predicted class label
class_idx = np.argmax(pred)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_name = class_names[class_idx]


print("The predicted class label is:", class_name)



model50k = build_model50k()
#model50k.summary()
model50k.save('build_model50k.h5')

#img = Image.open("/test_image_airplane.ext")
img = Image.open("/test_image_airplane.ext")
img = img.resize((32, 32))

  # Convert the image to a numpy array
x = np.array(img)
x = x.reshape((1,) + x.shape)  # Add batch dimension

  # Load the trained model
model = keras.models.load_model("build_model50k.h5")

  # Make a prediction on the image
pred = model.predict(x)

  # Get the predicted class label
class_idx = np.argmax(pred)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_name = class_names[class_idx]


print("The predicted class label is:", class_name)
  


