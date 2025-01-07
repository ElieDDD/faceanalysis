import streamlit as st
import os
from PIL import Image
from deepface import DeepFace

# Define the folder path where your images are stored
folder_path = 'glics'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

# Add title or description above the images
st.markdown("### This is a row of faces with sentiment analysis")

# Define the number of columns in the grid
num_columns = 3  # You can change this value based on how many columns you want

# Create columns for the grid
columns = st.columns([1] * num_columns)

# Loop through the image files and display them in a grid
for i, image_file in enumerate(image_files):
    # Open each image
    image_path = os.path.join(folder_path, image_file)
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize image to 100x100 pixels
    
    # Apply emotion detection using DeepFace
    analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    dominant_emotion = analysis[0]['dominant_emotion']
    
    # Display the image and sentiment label
    col = columns[i % num_columns]  # Distribute images across columns
    col.image(img, caption=f"{image_file}\nSentiment: {dominant_emotion}", use_column_width=False)
