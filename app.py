import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# App එකේ මාතෘකාව
st.title("🍎 Fruit Classifier AI")
st.write("Upload a fruit image to predict its name.")

# ==========================================
# 1. MODEL & CONFIGURATION
# ==========================================
MODEL_PATH = 'student_mobilenetv2_transfer_learning.keras'

# Model එක හැමතිස්සෙම load වෙන එක නතර කරලා speed කරන්න cache කරනවා
@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found in the repository!")
        return None

model = load_my_model()

# ඔයාගේ Fruit Classes 10 මෙතන නිවැරදි පිළිවෙළට දාන්න
CLASS_NAMES = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_fruit(img_file):
    img = Image.open(img_file).convert('RGB')
    img = img.resize((160, 160)) 
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 🌟 පරණ img_array /= 255.0 මකලා මේ පේළිය දාන්න:
    img_array = preprocess_input(img_array) 

    predictions = model.predict(img_array)
    highest_match_index = np.argmax(predictions[0])
    
    fruit_name = CLASS_NAMES[highest_match_index]
    confidence = float(predictions[0][highest_match_index] * 100)
    
    return fruit_name, round(confidence, 2)
# ==========================================
# 2. STREAMLIT UI & UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Upload කරපු image එක screen එකේ පෙන්වීම
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    if model is not None:
        with st.spinner('Classifying... Please wait...'):
            try:
                fruit, accuracy = predict_fruit(uploaded_file)
                
                # ප්‍රතිඵලය ලස්සනට පෙන්වීම
                st.success(f"### Result: **{fruit}**")
                st.info(f"Accuracy: **{accuracy}%**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
