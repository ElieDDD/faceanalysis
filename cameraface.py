import streamlit as st
import os
from PIL import Image

# Define the folder path where your images are stored
folder_path = 'glics'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

# Loop through the image files and display them
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    img = Image.open(image_path)
    st.image(img, caption=image_file, use_column_width=True)
