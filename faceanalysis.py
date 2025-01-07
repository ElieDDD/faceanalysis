import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from PIL import Image

# Function to generate an image maximizing the activation of a given class
def generate_maximized_image(model, target_class, image_size=(224, 224), iterations=50, learning_rate=10.0):
    # Initialize a random image
    input_image = np.random.random((1, *image_size, 3)) * 20 + 128.0
    input_image = tf.Variable(input_image, dtype=tf.float32)

    # Precompute the target class tensor (one-hot encoding)
    target_class_tensor = tf.one_hot([target_class], model.output.shape[-1])

    # Set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            predictions = model(input_image)
            loss = -tf.reduce_mean(predictions * target_class_tensor)  # Negative of class activation to maximize

        grads = tape.gradient(loss, input_image)
        optimizer.apply_gradients([(grads, input_image)])

        # Clip the image to stay within valid pixel values
        input_image.assign(tf.clip_by_value(input_image, 0, 255))

        if i % 10 == 0:  # Show progress every 10 iterations
            st.write(f"Iteration {i}/{iterations} - Loss: {loss.numpy()}")

    # Return the final image as a numpy array
    final_image = input_image.numpy().squeeze()
    return final_image

# Streamlit app to visualize Activation Maximization
def main():
    st.title("Activation Maximization Visualization")

    st.sidebar.header("Parameters")
    target_class = st.sidebar.slider("Choose Target Class", 0, 999, 340)  # ResNet50 has 1000 classes
    iterations = st.sidebar.slider("Number of Iterations", 1, 200, 50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.1, 20.0, 10.0)

    # Load ResNet50 pre-trained on ImageNet
    model = ResNet50(weights="imagenet")

    st.write("Generating Maximized Image...")
    
    # Generate the maximized image
    maximized_image = generate_maximized_image(model, target_class, iterations=iterations, learning_rate=learning_rate)

    # Decode the prediction of the maximized image
    predictions = model.predict(np.expand_dims(maximized_image, axis=0))
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    st.write(f"Predicted Class: {decoded_predictions[0][1]} (ID: {decoded_predictions[0][0]})")
    st.write(f"Class Confidence: {decoded_predictions[0][2]:.2f}")

    # Display the maximized image
    plt.imshow(maximized_image.astype(np.uint8))
    plt.axis("off")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
