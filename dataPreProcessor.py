import os
import re
import numpy as np
import pandas as pd

np.printoptions(precision=3)

imageData = []
ageGroupsRanges = 10 # the range of ages grouped together in csv file.
genders = {0: "male", 1: "female"}
race = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

for img in os.listdir('./Dataset/UTKFace'):
    fileLabels = np.array(re.findall("[0-9]{1,8}", img))

    imageData.append({"fileName": img,"age": int(fileLabels[0]), "gender": int(fileLabels[1]), "race": int(fileLabels[2])})

imageData = pd.DataFrame(imageData)
imageData.to_csv('./Dataset/data.csv')
