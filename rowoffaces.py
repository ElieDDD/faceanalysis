#requirements.txt should be
#streamlit
#Pillow

import streamlit as st
import os
from PIL import Image
from PIL import Image, ImageFilter

# Add custom CSS to ensure content is not clipped or cut off
st.markdown("""
    <style>
        .block-container {
            padding-top: 10px;  /* Ensure space at the top */
        }
        .stImage {
            margin: 0px;
            padding: 0px;
        }
        .stApp {
            overflow: visible;  /* Allow normal overflow */
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI Forensics")
# Define the folder path where your images are stored
folder_path = 'glics'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

# Add title or description above the images
st.markdown("### This is a row of faces")

# Define the number of columns in the grid
num_columns = 12  # You can change this value based on how many columns you want

# Create columns for the grid
columns = st.columns([1] * num_columns)
for i, image_file in enumerate(image_files):
    # Open and resize each image to 100x100
    image_path = os.path.join(folder_path, image_file)
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize image to 100x100 pixels
     # Apply blur effect to the image
    #img = img.filter(ImageFilter.GaussianBlur(radius=5))  # Apply blur with a specified radius (adjust as needed)
    
    # Display the image in the appropriate column
    col = columns[i % num_columns]  # Distribute images across columns
    col.image(img, "", use_container_width=False)
