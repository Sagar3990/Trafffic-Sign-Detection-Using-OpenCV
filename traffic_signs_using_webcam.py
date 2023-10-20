# import os
import cv2
import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import to_categorical
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam
from keras.models import model_from_json


def getclassname(classno):
    if classno == 0:
        return "speed limit 20 km/h"
    elif classno == 1:
        return "speed limit 30 km/h"
    elif classno == 2:
        return "speed limit 50 km/h"
    elif classno == 3:
        return "speed limit 60 km/h"
    elif classno == 4:
        return "speed limit 70 km/h"
    elif classno == 5:
        return "speed limit 80 km/h"
    elif classno == 6:
        return "end of speed limit 30 km/h"
    elif classno == 7:
        return "speed limit 100 km/h"
    elif classno == 8:
        return "speed limit 120 km/h"
    elif classno == 9:
        return "No passing"
    elif classno == 10:
        return "No passing over vehicles over 3.5 metric tons"
    elif classno == 11:
        return "Right of highway at the next intersection"
    elif classno == 12:
        return "Priority Road"
    elif classno == 13:
        return "Yield"
    elif classno == 14:
        return "Stop"
    elif classno == 15:
        return "NO vehicles"
    elif classno == 16:
        return "Vehicles over 3.5 metric tons prohibited"
    elif classno == 17:
        return "No Entry"
    elif classno == 18:
        return "General Caution"
    elif classno == 19:
        return "Dangerous Curve to the left"
    elif classno == 20:
        return "Dangerous Curve to the right"
    elif classno == 21:
        return "Double Curve"
    elif classno == 22:
        return "Bumpy Road"
    elif classno == 23:
        return "Slipery Road"
    elif classno == 24:
        return "Road narrows to the right"
    elif classno == 25:
        return "Road Work"
    elif classno == 26:
        return "Traffic Signals"
    elif classno == 27:
        return "Pedestrians"
    elif classno == 28:
        return "Children Crossing"
    elif classno == 29:
        return "Bicycles Crossing"
    elif classno == 30:
        return "Beware of Ice/Snow"
    elif classno == 31:
        return " Wild animals crossing"
    elif classno == 32:
        return "End of all speed and passing limits"
    elif classno == 33:
        return "Turn Right ahead"
    elif classno == 34:
        return "Turn Left ahead"
    elif classno == 35:
        return "Ahead only"
    elif classno == 36:
        return "Go straight or right"
    elif classno == 37:
        return "Go straight or left"
    elif classno == 38:
        return "Keep Right"
    elif classno == 39:
        return "Keep Left"
    elif classno == 40:
        return "Round about Mandatory"
    elif classno == 41:
        return "End of no passing"
    elif classno == 42:
        return "End of no passing vehicles over 3.5 metric tons"


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image/255
    return image


abc = open("traffic_signs.json", "r")
loaded_data = abc.read()
loaded_model = model_from_json(loaded_data)
loaded_model.load_weights("traffic_signs_weights.h5")

capt = cv2.VideoCapture(0)
capt.set(3, 640)
capt.set(4, 480)
capt.set(10, 180)
while True:
    message, image = capt.read()
    image_arr = np.asarray(image)
    image_arr = cv2.resize(image_arr, (32, 32))
    image_arr = preprocessing(image_arr)
    image_arr = image_arr.reshape(1, 32, 32, 1)
    predictions = loaded_model.predict(image_arr)
    Neuron_index = np.argmax(predictions, axis=1)
    # Neuron_index = loaded_model.predict_classes(image_arr)
    cv2.putText(image, "Class: ", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "Probability: ", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    probability_value = np.amax(predictions)
    if probability_value > 0.75:
        cv2.putText(image, getclassname(Neuron_index), (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(int(probability_value*100)) + "%", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Model Prediction", image)
    ascii_value = cv2.waitKey(1)
    if ascii_value == ord("q"):
        cv2.destroyAllWindows()
        break
