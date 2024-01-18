from keras.models import load_model
import numpy as np
import os
import cv2
import dlib

gender_model = load_model('./gender_predictor_model.h5')
age_model = load_model('./Models/agePredictionModel.keras')

detector = dlib.get_frontal_face_detector()

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
gender_dict = {0: "Male", 1: "Female"}

images_path = "./Dataset/"
image_files = [f for f in os.listdir("./Dataset/") if f.lower().endswith(('.jpg')) and os.path.isfile(os.path.join("./Dataset/", f))]

for image_file in image_files:
    print(images_path + image_file)
    image_path = os.path.join(images_path, image_file)
    image = cv2.imread(image_path)
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
        face_img = np.array(resized)
        face_to_predict = np.expand_dims(face_img ,axis=0)

        # make predictions
        gender_result = gender_model.predict(face_to_predict)
        age_result = age_model.predict(face_to_predict)


        compiled_age_result = {}

        for index, probability in enumerate(age_result[0]):
            compiled_age_result[ageGroups[index]] = int(probability * 100)

        pred_gender = gender_dict[round(gender_result[0][0])]

        # show prediction results
        print(f"Image: {image_file}, Predicted Gender: {pred_gender}")
        print(f"Image: {image_file}, Predicted ages: {compiled_age_result}")
