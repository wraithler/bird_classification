import librosa
import numpy as np

def extract_mfcc(file, sr=44100, n_mfcc=13):
    """
    Extracts MFCC features from an audio file.

    Args:
        file: Uploaded file-like object (wav).
        sr: Sample rate (default 44100).
        n_mfcc: Number of MFCC features to extract.

    Returns:
        A 1D numpy array of mean MFCC values.
    """
    y, sr = librosa.load(file, sr=sr, res_type='kaiser_best')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean
