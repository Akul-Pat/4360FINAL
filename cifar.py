import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Load and preprocess CIFAR-10 data
def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test

# Split data
X_train, X_test, y_train, y_test = load_and_preprocess_data()


# Define a simple CNN model
def create_classification_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Create the model
model = create_classification_model()

#number of epochs
epochs = 8

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2)


from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_true_classes, y_pred_classes)
print("Classification Report:")
print(class_report)

# Identify difficult-to-classify images based on prediction confidence
confidence_scores = np.max(y_pred, axis=1)
difficult_indices = np.argsort(np.abs(confidence_scores - 0.5))[:5]


# Visualize difficult images
for idx in difficult_indices:
    plt.imshow(X_test[idx])
    plt.title(f"True Class: {y_true_classes[idx]}, Predicted Class: {y_pred_classes[idx]}")
    plt.show()


def analyze_difficult_images(X_test, y_test, difficult_indices, model):
    for layer in model.layers:
        if 'conv2d' in layer.name:
            # Visualize for each convolutional layer
            last_conv_layer = model.get_layer(layer.name)

            for idx in difficult_indices:
                # Visualize the original image
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(X_test[idx])
                plt.title(f"True Class: {y_true_classes[idx]}, Predicted Class: {y_pred_classes[idx]}")

                # Visualize activation map using Grad-CAM
                plt.subplot(1, 2, 2)
                grad_cam_image = generate_grad_cam(X_test[idx], model, last_conv_layer)
                plt.imshow(grad_cam_image, cmap='jet')
                plt.title(f"Grad-CAM - Layer: {layer.name}")

                plt.show()


def generate_grad_cam(img_array, model, last_conv_layer):
    # Expand dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Model to get the activation maps
    grad_model = Model(model.inputs, [last_conv_layer.output, model.output])

    # Compute the gradient of the top predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Global average pooling
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Linear combination of filters and their weights
    cam = np.dot(output, weights)

    # Resize the CAM to match the original image size
    cam = cv2.resize(cam, (32, 32), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)  # ReLU-like activation

    # Normalize between 0 and 1
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam


# Analyze difficult images
analyze_difficult_images(X_test, y_test, difficult_indices, model)
print('done')

