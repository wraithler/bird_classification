# Birdie - Bird Sound Classification App 🐦

This is a simple Streamlit web app that classifies bird species based on uploaded audio files.

## How to run
1. Clone the repository.
2. Create a virtual environment, activate and install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the app:
    ```
    streamlit run app.py
    ```

## Requirements
- Python 3.12
- Streamlit
- Librosa
- Scikit-learn
- Pandas
- Matplotlib

## Model Info

This project uses a pre-trained model adapted from the WildWav GitHub (https://github.com/SughoshKulkarni/WildWav) repository. The model was originally trained on a larger dataset and has been simplified to detect a limited set of bird species: Cardinal, Mourning Dove, Pigeon, and Blue Jay.
