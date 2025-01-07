import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

# Set the page title
st.title("Real-Time Webcam Analysis")

# Streamlit sidebar
st.sidebar.title("Settings")
show_webcam = st.sidebar.checkbox("Open Webcam", value=True)

# Function to analyze the frame using DeepFace
def analyze_frame(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        return analysis
    except Exception as e:
        return None

# Open webcam if the checkbox is checked
if show_webcam:
    # Access the webcam
    video_capture = cv2.VideoCapture(0)

    # Display the webcam feed
    st.text("Press 'Stop' in Streamlit UI to stop the webcam.")
    stframe = st.empty()

    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            st.error("Error accessing webcam!")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame
        analysis = analyze_frame(rgb_frame)

        if analysis:
            # Display analysis results
            st.sidebar.write("**Analysis Results**")
            st.sidebar.write(f"Age: {analysis['age']}")
            st.sidebar.write(f"Gender: {analysis['gender']}")
            st.sidebar.write(f"Race: {analysis['dominant_race']}")
            st.sidebar.write(f"Emotion: {analysis['dominant_emotion']}")

            # Add text overlay to the frame
            cv2.putText(frame, f"Age: {analysis['age']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {analysis['gender']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Race: {analysis['dominant_race']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {analysis['dominant_emotion']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR")

    # Release the webcam
    video_capture.release()
    cv2.destroyAllWindows()
else:
    st.warning("Webcam is not enabled. Check the sidebar to enable it.")
