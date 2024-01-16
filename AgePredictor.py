from tqdm import tqdm
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#creating tqdm methods for pandas
tqdm.pandas()

# load dataset
print("loading dataset.")
dataset = pd.read_csv("./Dataset/data.csv")

# Load image data
IMAGE_SHAPE = (128, 128, 3)

print("loading image data.")
image_data = []

for index, datarow in tqdm(dataset.iterrows(), total=len(dataset)):
    image = load_img(datarow['filePath'], target_size=(128,128))
    image = img_to_array(image)
    image_data.insert(index, image)
    # image_data.append(image)

dataset['imageData'] = image_data
print(len(dataset['imageData']))
# show age distrobution histogram.
print("Showing general statistics")
plt.hist(dataset['age'], bins=100)
plt.title("Age distrobution")
# plt.show()

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

def mapAgeToAgeGroup(row):
    if (row['age'] <= 5):
        return 0
    elif (row['age'] <= 10):
        return 1
    elif (row['age'] <= 15):
        return 2
    elif (row['age'] <= 20):
        return 3
    elif (row['age'] <= 30):
        return 4
    elif (row['age'] <= 40):
        return 5
    elif (row['age'] <= 50):
        return 6
    elif (row['age'] <= 60):
        return 7
    elif (row['age'] <= 70):
        return 8
    elif (row['age'] <= 80):
        return 9
    elif (row['age'] <= 90):
        return 10
    else:
        return 11

dataset['ageGroups'] = dataset.progress_apply(mapAgeToAgeGroup, axis=1)

# print(dataset.tail(100))
print("finished creating age groups labels")

plt.figure(2)
plt.hist(dataset['ageGroups'], bins=12)
# plt.show()

# splitting data into training and test set
trainSet, testSet = np.split(dataset, [int(.9*len(dataset))]) # splits the dataset into a training set and a test set with the ratio 90%/10%
print(len(trainSet))
print(len(testSet))
print(len(trainSet) + len(testSet))

# splitting sets into features and labels.
trainFeature = trainSet['imageData']
trainLabels = trainSet['ageGroups']

testFeature = testSet['imageData']
testLabels = testSet['ageGroups']

print(testLabels)
