import keras
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('Models/agePredictionModel.keras')

# dlib face detector
detector = dlib.get_frontal_face_detector()

image = cv2.imread('TestImages/Mathias-25.jpg')

# face model classifier and model for selec best photos
faces = detector(image, 1)

# try to get first detected face from all images in image
for result in faces:

    # try to cut face from image
    x = result.left()
    y = result.top()
    w = result.right()
    h = result.bottom()

    # resize image for compatible with nn inputs
    resized = cv2.resize(image[y:h,x:w], (128, 128))

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
