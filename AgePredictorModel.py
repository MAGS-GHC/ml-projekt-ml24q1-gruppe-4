from tqdm import tqdm
import keras
from keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D

#creating tqdm methods for pandas
tqdm.pandas()

# load dataset
print("loading dataset.")
dataset = pd.read_csv("./Dataset/data.csv", index_col=0).astype({'age': 'float32'})
dataset.drop(columns=['gender','race'], inplace=True)
print(dataset.head())


# show age distrobution histogram.
print("Showing general statistics")
plt.hist(dataset['age'], bins=100)
plt.title("Age distrobution")
plt.show()

# split age into predefined age groups.
print("Splitting ages into age groups")

ageGroups = {
    0: "1-5",
    1: "6-10",
    2: "11-15",
    3: "16-20",
    4: "21-30",
    5: "31-40",
    6: "41-50",
    7: "51-60",
    8: "61-70",
    9: "71-80",
    10: "81-90",
    11: "91+"
}

def mapAgeToAgeGroup(age):
    if (age <= 5):
        return 0
    elif (age <= 10):
        return 1
    elif (age <= 15):
        return 2
    elif (age <= 20):
        return 3
    elif (age <= 30):
        return 4
    elif (age <= 40):
        return 5
    elif (age <= 50):
        return 6
    elif (age <= 60):
        return 7
    elif (age <= 70):
        return 8
    elif (age <= 80):
        return 9
    elif (age <= 90):
        return 10
    else:
        return 11

dataset['ageGroups'] = dataset['age'].progress_apply(mapAgeToAgeGroup)
dataset['mappedAgeGroups'] = dataset["ageGroups"].map(ageGroups)

print("finished creating age groups labels")

# plt.figure(2)
# plt.hist(dataset['ageGroups'], bins=12)
# plt.show()

dataset.drop(columns=['age'], inplace=True)


# splitting data into training and test set
train, test = train_test_split(dataset, test_size=0.1, random_state=2)


# prep labels.
train_labels = train['ageGroups']
test_labels = test['ageGroups']

# prep featueres.
print("loading image data.")
IMAGE_SHAPE = (128, 128, 3)

train_features = []
test_features = []

print("loading train images")

for index, datarow in tqdm(train.iterrows(), total=len(train)):
    image = load_img(datarow['filePath'], target_size=(128,128))
    image = np.array(image)
    train_features.append(image)

train_features = np.array(train_features)

print("loading test images")

for index, datarow in tqdm(test.iterrows(), total=len(test)):
    image = load_img(datarow['filePath'], target_size=(128,128))
    image = np.array(image)
    test_features.append(image)

test_features = np.array(test_features)

###


print("length of trainFeature: " + str(len(train_features)))
print("length of trainLabels: " + str(len(train_labels)))
print("length of testFeature: " + str(len(test_features)))
print("length of testLabels: " + str(len(test_labels)))

print("shape of trainFeature: " + str(train_features.shape))
print("shape of trainLabels: " + str(train_labels.shape))
print("shape of testFeature: " + str(test_features.shape))
print("shape of testLabels: " + str(test_labels.shape))

# Defining the model
agePredictionModel = keras.models.Sequential()
agePredictionModel.add(Conv2D(input_shape=IMAGE_SHAPE, kernel_size=(3, 3), filters=64, activation='relu'))
agePredictionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
agePredictionModel.add(Conv2D(kernel_size=(3, 3), filters=128, activation='relu'))
agePredictionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
agePredictionModel.add(Conv2D(kernel_size=(3, 3), filters=256, activation='relu'))
agePredictionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
agePredictionModel.add(Conv2D(kernel_size=(3, 3), filters=512, activation='relu'))
agePredictionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
agePredictionModel.add(Flatten())
agePredictionModel.add(Dense(units=1024, activation='relu'))
agePredictionModel.add(Dropout(0.5))
agePredictionModel.add(Dense(units=512, activation='relu'))
agePredictionModel.add(Dropout(0.4))
agePredictionModel.add(Dense(units=256, activation='relu'))
agePredictionModel.add(Dense(units=128, activation='relu'))
agePredictionModel.add(Dropout(0.4))
agePredictionModel.add(Dense(units=12, activation=keras.activations.softmax, kernel_initializer=keras.initializers.VarianceScaling()))

# compiling the model
adam_optimizer = keras.optimizers.Adam(learning_rate=0.0005)

agePredictionModel.compile(
    optimizer=adam_optimizer,
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

print(agePredictionModel.summary())

# Training the model
model_history = agePredictionModel.fit(
    train_features,
    train_labels,
    epochs=10,
    validation_data=(test_features, test_labels)
)

agePredictionModel.save('Models/agePredictionModel.keras')

plt.subplot(1, 2, 1)
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(model_history.history['loss'], label='training set')
plt.plot(model_history.history['val_loss'], label='test set')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(model_history.history['accuracy'], label='training set')
plt.plot(model_history.history['val_accuracy'], label='test set')
plt.legend()
plt.show()
