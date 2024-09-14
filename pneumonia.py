import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import AUC


from PIL import Image
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import threading
from colorama import Fore

EPOCHS = 10
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CATEGORIES = 2 #NORMAL/PNEUMONIA
BATCH_SIZE = 32

class YourTestCallback(Callback):
    def __init__(self, test_data):
        super(YourTestCallback, self).__init__()
        self.x_test, self.y_test = test_data

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # Append the test loss and accuracy to the logs
        logs['test_loss'] = test_loss
        logs['test_accuracy'] = test_accuracy
    
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
            continue  # Skip non-directory files
        
        for image_file in os.listdir(category_dir):
            if image_file.startswith(".DS_Store"):
                continue  # Skip .DS_Store files

            image_path = os.path.join(category_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            if category == "NORMAL": label = 0
            else: label = 1

            images.append(image)
            labels.append(label)
            show_progress(c, size)
            c += 1

    end_time = time.time()
    print(f"\nTime to Load Data: {round(end_time - start_time, 2)}s")

    count = {0: 0, 1: 0, 2: 0}
    for label in labels:
        count[label] += 1

    print("\n-=-=-=-=-DATA LOADED-=-=-=-=-")
    print(f"Normal:{count[0]}, Pneumonia:{count[1]}\n")
    return images, labels

# def create_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(NUM_CATEGORIES, activation='sigmoid')  # Binary classification
#     ])
    
#     optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def main():
    os.system("clear")

    if len(sys.argv) not in [1,2]:
        sys.exit("Usage: python3 model_pneumonia.py [save].model")
    elif len(sys.argv) == 2:
        print("Saving Model")
        
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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS, verbose = 1,
                      callbacks=YourTestCallback(test_data=(x_test, y_test)))

    model.evaluate(x_test, y_test, verbose=2)
    plot_history(history)

    #save (optional)
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")

def plot_history(history):
    test_accuracy = history.history['test_accuracy']
    test_loss = history.history['test_loss']

    epochs_range = range(1, EPOCHS + 1)  # Correct range of epoch numbers

    ## Plot accuracy
    plt.figure()  # Create a new figure for accuracy plots
    plt.plot(epochs_range, test_accuracy, 'b', label='Test accuracy')
    plt.title('Test Accuracy')
    plt.legend(loc=0)
    plt.show()

    ## Plot Loss
    plt.figure()  # Create a new figure for loss plots
    plt.plot(epochs_range, test_loss, 'b', label='Test Loss')
    plt.title('Test Loss')
    plt.legend(loc=0)
    plt.show()

if __name__ == "__main__":
    main()
