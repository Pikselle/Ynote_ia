#Automatiser les extraction et stocker les résultats

from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import librosa
import numpy as np
import soundfile as sf
import librosa.display
from fonctions import store_url
#choisir le fichier de test
fichier_audio= r"C:\Users\Theo\PycharmProjects\YNote\Ynote_ia\YNOTE\sample\Rick Astley - Never Gonna Give You Up (Official Music Video).mp3"
y, sr = librosa.load(fichier_audio)
# Partie des chromas features

def extract_chromas():


    # Compute the Chroma Short-Time Fourier Transform (chroma_stft)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # Calculate the mean chroma feature across time
    mean_chroma = np.mean(chromagram, axis=1)

# Define the mapping of chroma features to keys
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Find the key by selecting the maximum chroma feature
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]

# Print the detected key
    #print("Detected Key:", estimated_key)

    X, sample_rate = sf.read(fichier_audio, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    #print(chroma)
    return chroma
# Partie  MFCC

def extract_MFCC():
    # Calcul des MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return y, sr, mfcc

#Partie BPM
def extract_tempo():

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = tempo[0]
    bpm = int(round(tempo))
    return bpm

#chroma = extract_chromas()
#y, sr, mfcc = extract_MFCC()
#bpm = extract_tempo()
#print(chroma)
#print(bpm)
