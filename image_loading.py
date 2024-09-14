import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import time
import concurrent.futures
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CATEGORIES = 3  # NORMAL/BACTERIA PNEUMONIA/VIRAL PNEUMONIA
TEST_SIZE = 0.4

NUM_THREADS = 4  # Number of threads for parallel loading

def main():
    os.system('clear')
    dir_test = "../Pneumonia ML/DATA_chest_xray/test"
    dir_train = "../Pneumonia ML/DATA_chest_xray/train"
    images, labels = load_data(dir_train)

def load_images(image_files, category_dir):
    loaded_images = []
    for image_file in image_files:
        image_path = os.path.join(category_dir, image_file) 
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        loaded_images.append(image)
    return loaded_images

def load_data(data_dir):
    start_time = time.time()
    print("Loading Data...")
    images = []
    labels = []
    
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        
        if not os.path.isdir(category_dir):
            continue  # Skip non-directory files like DS-STORE
        
        image_files = [file for file in os.listdir(category_dir) if not file.startswith(".")]
        num_images = len(image_files)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            chunk_size = num_images // NUM_THREADS
            chunks = [image_files[i:i + chunk_size] for i in range(0, num_images, chunk_size)]
            results = []
            for chunk in chunks:
                result = executor.submit(load_images, chunk, category_dir)
                results.append(result)

            for result in concurrent.futures.as_completed(results):
                loaded_images = result.result()
                images.extend(loaded_images)
                if "NORMAL" in category_dir:
                    labels.extend([0] * len(loaded_images))
                elif "VIRAL" in category_dir:
                    labels.extend([1] * len(loaded_images))
                else:
                    labels.extend([2] * len(loaded_images))

    end_time = time.time()
    print(f"Time to Load Data: {round(end_time - start_time, 2)}s")

    label_counts = {0: 0, 1: 0, 2: 0}
    for label in labels:
        label_counts[label] += 1

    print(f"-=-=-=-=-=-=-=-DATA LOADED-=-=-=-=-=-=-=-\nNormal: {label_counts[0]}, Viral: {label_counts[1]}, Bacterial: {label_counts[2]}")
    return images, labels

def create_model():

    model = tf.keras.models.Sequential([
    #first convolution and pooling
    tf.keras.layers.Conv2D(128, (7, 7), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #second convolution and pooling
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])



    # Train neural network
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) 

    return model
if __name__ == "__main__":
    main()
