import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import RMSprop  # Import the legacy optimizer
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import AUC

from PIL import Image
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import threading
from colorama import Fore

EPOCHS = 25
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CATEGORIES = 3 #NORMAL/BACTERIA PNEUMONIA/VIRAL PNEUMONIA

def show_progress(current_image, total_images):
    progress = current_image / total_images
    bar_length = 40
    block = int(round(bar_length * progress))
    
    progress_bar = "[" + Fore.GREEN + "░" * block + Fore.WHITE + "-" * (bar_length - block) + "]"
    sys.stdout.write(f"\rLoading: {progress_bar} {progress*100:.2f}%")
    sys.stdout.flush()

def load_data(dir, data_dir, size):
    c = 1
    start_time = time.time()
    print("Loading data from", dir)

    images = []
    labels = []
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)

        if not os.path.isdir(category_dir):
            continue 
        
        for image_file in os.listdir(category_dir):
            if image_file.startswith(".DS_Store"):
                continue  

            image_path = os.path.join(category_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            if category == "NORMAL": label = 0
            elif category == "VIRAL": label = 1
            else: label = 2

            images.append(image)
            labels.append(label)
            show_progress(c, size)
            c += 1

    end_time = time.time()
    print(f"\nTime to Load Data: {round(end_time - start_time, 2)}s")

    count = {0: 0, 1: 0, 2: 0}
    for label in labels:
        count[label] += 1

    print("\n-=-=-=-=-=-=-=DATA LOADED=-=-=-=-=-=-=-")
    print(f"Normal:{count[0]}, Viral:{count[1]}, Bacterial:{count[2]}\n")
    return images, labels

def create_model():
    model = tf.keras.models.Sequential([
        # Layer 1: 2DConv with 256 filters and 3x3 kernel
        tf.keras.layers.Conv2D(256, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Activation('relu'),

        # Layer 2: MaxPooling with 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Layer 3: BatchNormalization
        tf.keras.layers.BatchNormalization(axis=1),

        # Layer 4: 2DConv with 64 filters and 3x3 kernel
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation('relu'),

        # Layer 5: MaxPooling with 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Layer 6: BatchNormalization
        tf.keras.layers.BatchNormalization(axis=1),

        # Layer 7: 2DConv with 16 filters and 3x3 kernel
        tf.keras.layers.Conv2D(16, (3, 3)),
        tf.keras.layers.Activation('relu'),

        # Flatten the output of the last convolutional layer
        tf.keras.layers.Flatten(),

        # Layer 8: Dense layer
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout layer with 0.5 dropout rate

        # Layer 9: Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    
    if len(sys.argv) not in [1,2]:
        sys.exit("Usage: python3 model_pneumonia.py [save].model")
    elif len(sys.argv) == 2:
        print("Creating and Saving Model")

    os.system('clear')
    dir_test = "../Pneumonia ML/DATA_chest_xray/test"
    dir_train = "../Pneumonia ML/DATA_chest_xray/train"

    #data prep
    images1, labels1 = load_data("training data", dir_train, 5232)
    labels1 = tf.keras.utils.to_categorical(labels1, NUM_CATEGORIES)
    x_train, y_train = np.array(images1) / 255.0, np.array(labels1)

    images2, labels2 = load_data("testing data" ,dir_test, 624)
    labels2 = tf.keras.utils.to_categorical(labels2, NUM_CATEGORIES)
    x_test, y_test = np.array(images2) / 255.0, np.array(labels2)

    #model
    model = create_model()
    #dropout_callback = AdjustDropoutCallback()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=2, validation_data=(x_val, y_val))
    evaluation = model.evaluate(x_test, y_test, verbose=2)
    plot_history(history)

    #save (optional)
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    ## Plot accuracy
    plt.plot(EPOCHS, acc, 'r', label='Training accuracy')
    plt.plot(EPOCHS, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    ## Plot Loss
    plt.plot(EPOCHS, loss, 'r', label='Training Loss')
    plt.plot(EPOCHS, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

if __name__ == "__main__":
    main()
