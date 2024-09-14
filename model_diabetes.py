import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from PIL import Image
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import threading
from colorama import Fore

EPOCHS = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CATEGORIES = 4 
BATCH_SIZE = 32

def main():
    if len(sys.argv) not in [1,2]:
        sys.exit("Usage: python3 model_diabetes.py [save].model")
    elif len(sys.argv) == 2:
        print("Creating and Saving Model")

    os.system('clear')
    dir_test = "../Diabetes ML/DATA_eye_OCT/test"
    dir_train = "../Diabetes ML/DATA_eye_OCT/train"

    #data prep
    images1, labels1 = load_data("training data", dir_train, 32000) #total 108309
    labels1 = tf.keras.utils.to_categorical(labels1, NUM_CATEGORIES)
    x_train, y_train = np.array(images1) / 255.0 , np.array(labels1)

    images2, labels2 = load_data("testing data" ,dir_test, 1000)
    labels2 = tf.keras.utils.to_categorical(labels2, NUM_CATEGORIES)
    x_test, y_test = np.array(images2) / 255.0, np.array(labels2)

    #model
    model = create_model()
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test, verbose = 2)
    
    #save (optional)
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    # Plot accuracy
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    # Plot Loss
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

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

            if category == "CNV": 
                label = 1
            elif category == "DME": 
                label = 2
            elif category == "DRUSEN": 
                label = 3
            else: label = 0 #normal

            images.append(image)
            labels.append(label)
            show_progress(c, size)
            c += 1
            if c % 2000 == 0:
                break

    end_time = time.time()
    print(f"\nTime to Load Data: {round(end_time - start_time, 2)}s")

    count = {0: 0, 1: 0, 2: 0, 3: 0}
    for label in labels:
        count[label] += 1

    print("\n-=-=-=-=-=-=-=DATA LOADED=-=-=-=-=-=-=-")
    print(f"Normal:{count[0]}, CNV:{count[1]}, DME:{count[2]}, DRUSEN:{count[3]}\n")
    return images, labels

def create_model():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 180x180 with 3 bytes colour
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), 
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(512, activation='relu'), 
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')  
    ])

    model.compile(optimizer = 'adam', 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    main()



# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # ... (other code)

# def main():
#     # ... (other code)
    
#     # Create data generators for training and validation
#     train_datagen = ImageDataGenerator(
#         rescale=1.0/255.0,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
    
#     test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
#     train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
#     test_generator = test_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE)
    
#     # ...

# def load_data(dir, data_dir, size):
#     # ... (other code)
    
#     # Use a generator to yield images and labels in batches
#     data_generator = ImageDataGenerator(rescale=1.0/255.0)
#     data_iterator = data_generator.flow(np.array(images), np.array(labels), batch_size=BATCH_SIZE)
    
#     # Fetch data in batches
#     batch_images, batch_labels = data_iterator.next()
    
#     for i in range(size):
#         images.append(batch_images[i])
#         labels.append(batch_labels[i])
#         show_progress(c, size)
#         c += 1
#         if c >= size:
#             break

#     # ...

# if __name__ == "__main__":
#     main()
