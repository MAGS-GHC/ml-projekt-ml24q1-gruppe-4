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

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPU devices available.")
else:
    print("GPU devices available:", physical_devices)

# Specify the correct directory path
path = Path("./Dataset/UTKFace")
#path = Path(r"C:\Users\Niklas Pedersen\Downloads\UTKFace")

# Use glob to get all files with the '.jpg' extension
jpg_files = list(path.glob('*.jpg'))

# Extract file names
filenames = list(map(lambda x: x.name, jpg_files))

#print(len(filenames))
#print(filenames[:3])

np.random.seed(10)
np.random.shuffle(filenames)

gender_labels, image_path = [], []

for filename in filenames:
    image_path.append(filename)
    temp = filename.split('_')
    gender_labels.append(temp[1])

df = pd.DataFrame()
df['image'], df['gender'] = image_path, gender_labels

print(df.head())

gender_dict = {0:"Male",1:"Female"}
df = df.astype({'gender': 'int32'})
print(df.dtypes)

#img = Image.open("./Dataset/UTKFace/" + df.image[1])
#plt.imshow(img)
#plt.show()

image_path = "./Dataset/UTKFace/"
'''
files = df.iloc[0:20]
plt.figure(figsize=(15,15))
for index, file, gender in files.itertuples():
    plt.subplot(5,5, index+1)
    img = load_img(image_path+file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Gender: {gender_dict[gender]}")
    plt.axis('off')

plt.show()
'''

train, test = train_test_split(df, test_size=0.20, random_state=42)
print(train.head())

x_train = []

for file in train.image:
    img = load_img(image_path + file, target_size=(128, 128))
    img = np.array(img)  # Convert PIL Image to numpy array
    x_train.append(img)

x_train = np.array(x_train)

print(x_train.shape)

x_test = []

for file in test.image:
    img = load_img(image_path + file, target_size=(128, 128))
    img = np.array(img)  # Convert PIL Image to numpy array
    x_test.append(img)

x_test = np.array(x_test)

print(x_test.shape)

x_train = x_train/255
x_test = x_test/255

y_gender = np.array(train.gender)
print(len(y_gender))

y_test_gender = np.array(test.gender)
print(len(y_test_gender))

input_size = (128,128,3)

inputs = Input((input_size))
X = Conv2D(64, (3, 3), activation='relu', kernel_initializer = glorot_uniform(seed=0))(inputs)
X = BatchNormalization(axis = 3)(X)
X = MaxPooling2D((3, 3))(X)

X = Conv2D(128, (3, 3), activation='relu')(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)

X = Conv2D(256, (3, 3), activation='relu')(X)
X = MaxPooling2D((2, 2))(X)

X = Flatten()(X)

dense_1 = Dense(256, activation='relu')(X)
dense_2 = Dense(256, activation='relu' )(X)
dense_3 = Dense(128, activation='relu' )(dense_2)
dropout_1 = Dropout(0.4)(dense_1)
output_1 = Dense(1,activation='sigmoid', name='gender_output')(dropout_1)

model = Model(inputs=[inputs], outputs=[output_1])

'''
vgg_model=tf.keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3))

for layer in vgg_model.layers:
    layer.trainable = False
vgg_model.summary()

flatten_layer = Flatten()(vgg_model.output)

dense_1 = Dense(256, activation='relu')(flatten_layer)
dense_2 = Dense(256, activation='relu')(flatten_layer)
dense_3 = Dense(128, activation='relu')(dense_2)

dropout_1 = Dropout(0.4)(dense_1)

output_1 = Dense(1, activation='sigmoid', name='gender_output')(dropout_1)

# Create the final model
vgg_model = Model(inputs=vgg_model.input, outputs=[output_1, output_2])

vgg_model.summary()
'''

model.compile(loss=['binary_crossentropy','mae'], optimizer='adam', metrics=['accuracy'])

#vgg_model.compile(loss=['binary_crossentropy','mae'], optimizer='adam', metrics=['accuracy'])

#vgg_model.summary()

model.summary()

num_epochs = 20
batch_size = 32
steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_test) // batch_size

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

train_generator = train_datagen.flow(x_train, y_gender, batch_size=batch_size)
test_generator = test_datagen.flow(x_test, y_test_gender, batch_size=batch_size)

model_history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps
)
model_path = "./Dataset/UTKFace/gender_predictor_model.h5"
model.save(model_path)

'''
vgg_model_history = vgg_model.fit_generator(
    generator=multi_output_flow_train,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=multi_output_flow_test,
    validation_steps=len(x_test)
)
'''

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Gender loss cnn')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

'''
plt.plot(vgg_model_history.history['gender_output_loss'])
plt.plot(vgg_model_history.history['val_gender_output_loss'])
plt.title('Gender loss vgg')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
'''