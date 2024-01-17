import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.layers import Dropout, Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = "./gender_predictor_model.h5"
model = load_model(model_path)

gender_dict = {0: "Male", 1: "Female"}

images_path = "./Dataset/"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg')) and os.path.isfile(os.path.join(images_path, f))]

for image_file in image_files:
    # Load and preprocess each image
    image_path = os.path.join(images_path, image_file)
    new_img = Image.open(image_path)
    new_img = new_img.resize((128, 128))  # Resize the image to match the input size
    new_img_array = np.array(new_img)
    new_img_array = np.expand_dims(new_img_array, axis=0)
    new_img_array = new_img_array / 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(new_img_array)

    pred_gender = gender_dict[round(predictions[0][0])]

    print(f"Image: {image_file}, Predicted Gender: {pred_gender}")
    print("\n")