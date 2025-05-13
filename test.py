import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import requests
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
from bs4 import BeautifulSoup
import time
filterwarnings('ignore')

# ----------------- Gemini API Setup --------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
API_KEY = "AIzaSyBic9wv9KVhgehOhbT3ufe4g_kG1laSdbQ"  # 🔒 Replace with your actual key or use os.getenv("GEMINI_API_KEY")

# Typewriter Effect Function
def typewriter_effect(text, delay=0.01):
    placeholder = st.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(delay)

# Get bird info from Gemini API
def get_bird_info(bird_name):
    """Get 2-line structured bird info using Gemini API"""
    prompt = f"""
    Provide concise information about the bird: {bird_name}
    
    Use the following structure with **each section limited to 2 lines only**:
    1. Scientific Name
    2. Common Name
    3. Physical Description (size, color, beak, feet, distinct features)
    4. Habitat (geographical range, preferred environment)
    5. Behavior (diet, feeding, social behavior, communication)
    6. Reproduction (mating season, eggs, chick development)
    7. Lifespan
    8. Conservation Status (IUCN Red List status, major threats)
    9. Cultural Significance (symbolism, myths, human interaction)
    10. Interesting Facts (adaptations, famous individuals, notable traits)

    Keep each section informative and to the point.
    """

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    params = {"key": API_KEY}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, params=params)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error getting response: {str(e)}"

# ----------------- Scraping Bird Image from Wikipedia --------------------
def fetch_bird_image(bird_name):
    """Fetch bird image from Wikipedia"""
    url = f"https://en.wikipedia.org/wiki/{bird_name.replace(' ', '_')}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we get a successful response
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tag = soup.find('img')
        
        # Construct the image URL
        image_url = "https:" + image_tag['src']
        return image_url
    except Exception as e:
        return None  # Return None if the image could not be fetched

# ----------------- Streamlit Page Setup --------------------
def streamlit_config():
    st.set_page_config(page_title='Bird Sound Classification', layout='centered')

    # Transparent header style
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # Title
    st.markdown(f'<h1 style="text-align: center;">Bird Sound Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)

streamlit_config()

# ----------------- Prediction Function --------------------
def prediction(audio_file):
    with open('prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    audio, sample_rate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    model = tf.keras.models.load_model('final.h5')
    prediction = model.predict(mfccs_tensors)
    target_label = np.argmax(prediction)
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction) * 100, 2)

    add_vertical_space(1)
    st.markdown(f'<h4 style="text-align: center; color: orange;">{confidence}% Match Found</h4>', 
                unsafe_allow_html=True)

    # ✅ Load image only from local folder
    image_folder = 'Inference_Images'
    jpg_path = os.path.join(image_folder, f'{predicted_class}.jpg')
    jpeg_path = os.path.join(image_folder, f'{predicted_class}.jpeg')

    if os.path.exists(jpg_path):
        image_path = jpg_path
    elif os.path.exists(jpeg_path):
        image_path = jpeg_path
    else:
        image_path = None

    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (350, 300))   
        _, col2, _ = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.image(img, caption=predicted_class, use_container_width=True)
    else:
        st.warning("📁 No local image found (.jpg or .jpeg) for this bird.")

    st.markdown(f'<h3 style="text-align: center; color: green;">{predicted_class}</h3>', 
                unsafe_allow_html=True)

    # 🔍 Bird info from Gemini
    st.markdown("#### 🔎 Bird Information :")
    with st.spinner("Getting bird info..."):
        gemini_response = get_bird_info(predicted_class)
        typewriter_effect(gemini_response, delay=0.01)


# ----------------- Streamlit UI --------------------
_, col2, _ = st.columns([0.1, 0.9, 0.1])
with col2:
    input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])

if input_audio is not None:
    _, col2, _ = st.columns([0.2, 0.8, 0.2])
    with col2:
        prediction(input_audio)
