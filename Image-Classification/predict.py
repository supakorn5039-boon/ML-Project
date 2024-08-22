import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Load the trained model
model = models.load_model('image_classifier.keras')

# Load and preprocess the image
img = cv.imread('cat.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_resized = cv.resize(img, (32, 32))  # Resize image to match model input

# Prepare the image for prediction
img_array = np.array([img_resized]) / 255.0

# Make prediction
prediction = model.predict(img_array)
index = np.argmax(prediction)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
predicted_class = class_names[index]

# Display the image and prediction
plt.imshow(img_resized)
plt.title(f'Prediction: {predicted_class}')
plt.axis('off')  # Hide axes
plt.show()

print(f'Prediction is: {predicted_class}')
