import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import json
from audio_processing.features import extract_mfcc

# loading model and bird information 
with open('model/bird_predictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('assets/bird_images.json', 'r') as bird_file:
    bird_data = json.load(bird_file)

labels = [f"var{i}" for i in range(1, 14)]  # expected MFCC features

# setting the page on streamlit
st.set_page_config(page_title="Bird Sound Classifier", page_icon="🐦")
st.title("🐤 Bird Sound Identification App")
st.subheader("Upload a bird sound file (.wav) and find out which bird it is!")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    try:
        # doing the feature extraction
        mfcc_features = extract_mfcc(uploaded_file)
        input_features = pd.DataFrame([mfcc_features], columns=labels)

        # predicting the bird
        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)
        confidence = np.max(probabilities)

        if confidence > 0.7:
            st.success(f"✅ It's a **{prediction}**! (Confidence: {confidence*100:.2f}%)")
            st.image(bird_data[prediction]["image"], caption=prediction)
            st.markdown(f"[🔗 Learn more about {prediction}]({bird_data[prediction]['link']})")
        else:
            st.warning("⚠️ Hmm, the model isn't confident enough to recognize this bird. Try a clearer recording!")

    except Exception as e:
        st.error(f"❌ Error processing the audio file: {e}")

else:
    st.info("👈 Upload a WAV file to get started!")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
