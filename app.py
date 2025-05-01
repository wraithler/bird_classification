import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import json
from audio_processing.features import extract_mfcc
from PIL import Image
import time

st.set_page_config(
    page_title="Birdie - Bird Sound Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #4CAF50;
        padding: 1rem 2rem;
        color: white;
        width: 100%;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .navbar-brand {
        font-size: 2rem;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .navbar-brand-icon {
        margin-right: 0.5rem;
        font-size: 2.2rem;
    }
    .navbar-links {
        display: flex;
        gap: 2rem;
    }
    .navbar-link {
        color: white;
        text-decoration: none;
        font-size: 1.1rem;
    }
    .navbar-link:hover {
        text-decoration: underline;
    }
    .main-content {
        margin-top: 5rem;
        padding: 1rem;
    }
    .sub-header {
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .bird-info {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f4f8;
    }
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #7f8c8d;
    }
    #MainMenu, header, footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 0rem;
        max-width: 100%;
    }
    .css-1d391kg, .css-1lcbmhc {
        padding-top: 5rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("About the App")
    st.markdown("""
    This app uses machine learning to identify bird species from audio recordings.
    
    ### How to use:
    1. Upload a WAV file of bird sounds
    2. Wait for the analysis
    3. Get the predicted bird species
    
    ### Features:
    - Fast identification
    - High accuracy for common species
    - Educational information about identified birds
    """)
    
    st.markdown("---")
    st.caption("© 2025 Bird Sound Identifier")

@st.cache_resource
def load_model():
    with open('model/bird_predictor.pkl', 'rb') as model_file:
        return pickle.load(model_file)

@st.cache_data
def load_bird_data():
    with open('assets/bird_images.json', 'r') as bird_file:
        return json.load(bird_file)

model = load_model()
bird_data = load_bird_data()
labels = [f"var{i}" for i in range(1, 14)]

st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span class="navbar-brand-icon">🐦</span> Birdie
    </div>
    <div class="navbar-links">
        <a href="#" class="navbar-link">Home</a>
        <a href="#" class="navbar-link">About</a>
        <a href="#" class="navbar-link">Gallery</a>
        <a href="#" class="navbar-link">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a bird sound and discover which species it belongs to</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a WAV file of bird sounds", 
        type=["wav"],
        help="For best results, use clear recordings with minimal background noise"
    )

with col2:
    if not uploaded_file:
        st.markdown("""
        **Tips for good recordings:**
        - Record in early morning when birds are most active
        - Try to get as close as possible to the bird
        - Minimize background noise
        """)

if uploaded_file:
    file_details = {"Filename": uploaded_file.name, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
    st.write("**File Details:**", file_details)
    
    st.markdown("**Listen to your recording:**")
    st.audio(uploaded_file, format='audio/wav')
    
    st.markdown("### Analysing bird sounds...")
    
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    try:
        mfcc_features = extract_mfcc(uploaded_file)
        input_features = pd.DataFrame([mfcc_features], columns=labels)

        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)
        confidence = np.max(probabilities)
        
        sorted_indices = np.argsort(probabilities[0])[::-1]
        top_predictions = [(model.classes_[i], probabilities[0][i]) for i in sorted_indices[:3]]
        
        st.markdown("## 🔍 Analysis Results")
        
        result_col1, result_col2 = st.columns([3, 2])
        
        with result_col1:
            if confidence > 0.7:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### ✅ We're confident this is a **{prediction}**!")
                st.markdown(f"**Confidence level:** {confidence*100:.1f}%")
                
                st.progress(confidence)
                
                st.markdown("#### Other possibilities:")
                for bird, prob in top_predictions[1:]:
                    st.markdown(f"- {bird}: {prob*100:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Hmm, the model isn't confident enough to recognize this bird. Try a clearer recording!")
                st.markdown("#### Possible matches:")
                for bird, prob in top_predictions:
                    st.markdown(f"- {bird}: {prob*100:.1f}%")
        
        with result_col2:
            if confidence > 0.7:
                st.markdown('<div class="bird-info">', unsafe_allow_html=True)
                st.image(bird_data[prediction]["image"], caption=prediction, use_column_width=True)
                
                st.markdown("#### About this bird:")
                if "description" in bird_data[prediction]:
                    st.markdown(bird_data[prediction]["description"])
                
                st.markdown(f"[🔗 Learn more about {prediction}]({bird_data[prediction]['link']})")
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing the audio file: {str(e)}")
        st.markdown("""
        **Possible issues:**
        - The file may be corrupted
        - The format might not be compatible
        - The recording might be too short or unclear
        
        Try uploading a different file or check if the WAV format is correct.
        """)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("---")
st.markdown("[Report Issues](https://github.com/) | [Source Code](https://github.com/)")
st.markdown('</div>', unsafe_allow_html=True)