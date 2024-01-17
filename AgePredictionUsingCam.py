import keras
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('Models/agePredictionModel.keras')

# dlib face detector
detector = dlib.get_frontal_face_detector()

# enable web cam
video_capture = cv2.VideoCapture(0)

# Capture frame
ret, frame = video_capture.read()

# face model classifier and model for selec best photos
faces = detector(frame, 1)

# try to get first detected face from all images in image
for result in faces:

    # try to cut face from image
    x = result.left()
    y = result.top()
    w = result.right()
    h = result.bottom()

    # resize image for compatible with nn inputs
    resized = cv2.resize(frame[y:h,x:w], (128, 128))

    # convert image from bytes to array with int
    face_img=np.array(resized)

    # preparing image array to one shape for neural network 
    face_to_predict=np.expand_dims(face_img ,axis=0)
    print(face_img.shape)
    plt.imshow(face_img)
    plt.show()
    # recognize image
    result=model.predict(face_to_predict)

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

    compiled_result = {}

    for index, probability in enumerate(result[0]):
        compiled_result[ageGroups[index]] = int(probability * 100)

    # show prediction results
    print(compiled_result)
    break;

{
    '1-5': 0.013345154, 
    '6-10': 0.012405647, 
    '11-15': 0.012025213, 
    '16-20': 0.02129515, 
    '21-30': 0.20825106, 
    '31-40': 0.25609604, 
    '41-50': 0.18747611, 
    '51-60': 0.1612157, 
    '61-70': 0.079934634, 
    '71-80': 0.032345187, 
    '81-90': 0.012966989, 
    '91+': 0.0026431372
 }

# [
#     0: 0.04033662, 
#     1: 0.01130453,
#     2: 0.00478719, 
#     3: 0.00975279, 
#     4: 0.15918605, 
#     5: 0.12693813, 
#     6: 0.12167366, 
#     7: 0.167016  , 
#     8: 0.1669102 , 
#     9: 0.10463162, 
#     10: 0.07787441, 
#     11: 0.00958882
# ]
