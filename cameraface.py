import streamlit as st
import os
from PIL import Image

# Add custom CSS to reduce spacing
st.markdown("""
    <style>
        .stImage {
            margin: 4px;
            padding: 0px;
        }
        .block-container {
            padding: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# Define the folder path where your images are stored
folder_path = 'glics'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

# Define the number of columns in the grid
num_columns = 10  # You can change this value based on how many columns you want

# Create columns for the grid
columns = st.columns([1] * num_columns)

# Loop through the image files and display them in a grid
for i, image_file in enumerate(image_files):
    # Open and resize each image to 100x100
    image_path = os.path.join(folder_path, image_file)
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize image to 100x100 pixels
    
    # Display the image in the appropriate column
    col = columns[i % num_columns]  # Distribute images across columns
    col.image(img, "", 100)
