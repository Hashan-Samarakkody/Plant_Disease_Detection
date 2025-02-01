import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import time

# Streamlit UI - set page config as the first streamlit command
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Configuration and Setup
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'past' not in st.session_state:
    st.session_state.past = []
if 'generated' not in st.session_state:
    st.session_state.generated = []

# Load the model with error handling
try:
    model = tf.keras.models.load_model(model_path)
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
except Exception as e:
    st.error("Error loading model or class indices. Please check file paths.")
    model = None
    class_indices = {}

# Configure Gemini AI
try:
    genai.configure(api_key="API_KEY")
    config = GenerationConfig(
        temperature=0.9,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024
    )
    chat_model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        generation_config=config
    )
except Exception as e:
    st.error("Error configuring Gemini AI. Please check your API key.")
    chat_model = None

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def predict_image_class(model, image_path, class_indices):
    if model is None:
        return "Cannot identify"
    try:
        preprocessed_img = load_and_preprocess_image(image_path)
        if preprocessed_img is None:
            return "Cannot identify"

        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Cannot identify")
        return predicted_class_name
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Cannot identify"

def get_chatbot_response(query, disease=None):
    if chat_model is None:
        return "Chat service is currently unavailable. Please check your API configuration."

    try:
        if disease:
            prompt = f"The plant disease detected is {disease}. Please provide detailed information about this disease, including its symptoms, causes, and treatment options. Keep the response concise and practical."
        else:
            prompt = query

        response = chat_model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            return response.text
        return "Unable to generate response"
    except Exception as e:
        return f"I apologize, but I'm having trouble generating a response: {str(e)}"

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    response = get_chatbot_response(user_input)
    st.session_state.generated.append(response)

def on_btn_click():
    st.session_state.past = []
    st.session_state.generated = []

# Background Animation and Styling
st.markdown("""
    <style>
        * {
            padding: 0;
            margin: 0;
            box-sizing: border-box;
        }

        :root {
            --dark-color: #000;
        }

        body {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            min-height: 100vh;
            background-color: var(--dark-color);
            overflow: hidden;
            perspective: 1000px;
        }

        .night {
            position: fixed;
            left: 50%;
            top: 0;
            transform: translateX(-50%);
            width: 100%;
            height: 100%;
            filter: blur(0.1vmin);
            background-image: radial-gradient(
                    ellipse at top,
                    transparent 0%,
                    var(--dark-color)
                ),
                radial-gradient(
                    ellipse at bottom,
                    var(--dark-color),
                    rgba(145, 233, 255, 0.2)
                );
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Plant Disease Detection & Assistant 🌿")
st.markdown("""
    Welcome to the Plant Disease Detection tool! Upload a leaf image to detect the disease and interact with the assistant for treatment advice.
    
    ### Steps:
    1. **Upload Image** 📸: Upload a clear image of a plant leaf.
    2. **Analyze** 🔍: Click on "Analyze Disease" to detect the disease.
    3. **Chat with Assistant** 💬: Ask questions about the disease's treatment and causes.
""")

# Image Upload and Disease Prediction
uploaded_image = st.file_uploader("Upload a plant leaf image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([2, 4])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        analyze_button = st.button('Analyze Disease')
        if analyze_button:
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.session_state.last_prediction = prediction

            if prediction != "Cannot identify":
                st.success(f'✅ Detected Disease: {prediction}!')
                response = get_chatbot_response("", disease=prediction)
                st.session_state.past.append(f"Detected Disease: {prediction}")
                st.session_state.generated.append(response)
            else:
                st.warning("⚠️ Could not identify the disease. Please try another image.")

# Chat Interface
chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i], is_user=True, key=f"{i}_user")
        message(st.session_state.generated[i], key=f"{i}_bot")

st.text_input("User Input:", on_change=on_input_change, key="user_input")
st.button("Clear Chat 🪮", on_click=on_btn_click)
