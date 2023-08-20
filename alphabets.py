from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import cv2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h))

def train_model(X_train, y_train, num_classes, epochs, alpha):
    encode = OneHotEncoder(sparse=False)
    Y = encode.fit_transform(y_train.reshape(-1, 1))

    X = X_train
    m = len(Y)
    B = np.zeros([num_classes, 64])

    for iteration in range(epochs):
        dB = np.zeros(B.shape)
        Loss = 0
        for j in range(X.shape[0]):
            x1 = X[j, :].reshape(64, 1)
            y1 = Y[j, :].reshape(num_classes, 1)

            z1 = np.dot(B, x1)
            h = sigmoid(z1)

            db = (h - y1) * x1.T

            dB += db
            Loss += cost_function(h, y1)

        dB = dB / float(X.shape[0])
        Loss = Loss / float(X.shape[0])
        gradient = alpha * dB
        B = B - gradient

    return B

def recognize_alphabet(image_path, B):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(255 - gray, (8, 8))

    g1 = gray.reshape(64, 1)
    z1 = np.dot(B, g1)
    h = sigmoid(z1)
    np.set_printoptions(suppress=True)

    # Return the index of the predicted class
    return h.argmax()

# Create a synthetic dataset for alphabets A, B, and C
num_samples_per_class = 100
num_classes = 3
alphabet_data = []

for _ in range(num_samples_per_class):
    letter_A = np.zeros((8, 8))
    letter_A[1:7, 2:6] = 1
    alphabet_data.append((letter_A.flatten(), 0))

for _ in range(num_samples_per_class):
    letter_B = np.zeros((8, 8))
    letter_B[1, 1:7] = 1
    letter_B[4, 1:7] = 1
    letter_B[7, 1:7] = 1
    letter_B[2:7, 1] = 1
    letter_B[2:7, 6] = 1
    alphabet_data.append((letter_B.flatten(), 1))

for _ in range(num_samples_per_class):
    letter_C = np.zeros((8, 8))
    letter_C[1, 1:7] = 1
    letter_C[7, 1:7] = 1
    letter_C[2:7, 1] = 1
    letter_C[2:7, 6] = 1
    alphabet_data.append((letter_C.flatten(), 2))

np.random.shuffle(alphabet_data)

X = np.array([data[0] for data in alphabet_data])
y = np.array([data[1] for data in alphabet_data])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

epochs = 100
alpha = 0.1

B = train_model(x_train, y_train, num_classes, epochs, alpha)

image_path = "Alphbets/A1.png" 
predicted_class = recognize_alphabet(image_path, B)

predicted_letter = chr(ord('A') + predicted_class)
print(f"Predicted letter: {predicted_letter}")