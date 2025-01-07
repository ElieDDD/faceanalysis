import streamlit as st
from deepface import DeepFace
import json

st.title("🎭 Facial Analysis Application")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img_path = uploaded_file.name

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'])
        
        important_info = {
            "Age": analysis[0]["age"],
            "Gender": max(analysis[0]["gender"] , key=analysis[0]["gender"].get) ,
            "Race": analysis[0]["dominant_race"],
            "Emotion": analysis[0]["dominant_emotion"]
        }

        st.subheader("🌈 Analysis Results:")
        for key, value in important_info.items():
            color = {
                "Age": "orange",
                "Gender": "orange",
                "Race": "orange",
                "Emotion": "orange"
            }.get(key, "black")
            st.markdown(f"<span style='color:{color}; font-size: 20px;'><strong>{key}:</strong> {value}</span>", unsafe_allow_html=True)

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("This application uses the DeepFace library to analyze facial data such as age, gender, race, and emotion.")
st.markdown("🌐 Developed by [Biplob D.](https://github.com/biplob004)")
