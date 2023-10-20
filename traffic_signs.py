import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import model_from_json


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image/255
    return image


features = []
targets = []
for i in list(range(0, 43)):
    collectionImageNames = os.listdir("D:/myData" + "/" + str(i))
    for j in collectionImageNames:
        img = cv2.imread("D:/myData" + "/" + str(i) + "/" + j)
        features.append(img)
        targets.append(i)
    print("Loading in folder", i)
features = np.array(features)
targets = np.array(targets)
train_features, test_features, train_target, test_target = train_test_split(features, targets, test_size=0.2)

train_features = np.array(list(map(preprocessing, train_features)))
test_features = np.array(list(map(preprocessing, test_features)))
print(train_features.shape)
print(test_features.shape)
train_features = train_features.reshape(27839, 32, 32, 1)
test_features = test_features.reshape(6960, 32, 32, 1)


DataGen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.1, zoom_range=0.2)
DataGen.fit(train_features)
batches = DataGen.flow(train_features, train_target, batch_size=20)
train_target = to_categorical(train_target)
# step 1------ Specify the architecture
model = Sequential()
model.add(Conv2D(60, (3, 3), activation="relu", input_shape=(32, 32, 1)))
model.add(Conv2D(60, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(30, (3, 3), activation="relu"))
model.add(Conv2D(30, (3, 3), activation="relu"))
model.add(Conv2D(30, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(43, activation="softmax"))

# Step 2----- Compile the model
model.compile(Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Step 3-----Train the model
model.fit(DataGen.flow(train_features, train_target, batch_size=20), epochs=20)
# saving the model architecture in the json file
Model_In_Json = model.to_json()
abc = open("traffic_signs.json", "w")
abc.write(Model_In_Json)
abc.close()
# saving the model weights in the h5 file
model.save_weights("traffic_signs_weights.h5")
# Reading(loading..) the json file to use it in the model
abc = open("traffic_signs.json", "r")
loaded_data = abc.read()
loaded_model = model_from_json(loaded_data)
loaded_model.load_weights("traffic_signs_weights.h5")
# Step 4------Test the model for the predictions
predictions = loaded_model.predict(test_features)
print(predictions)
