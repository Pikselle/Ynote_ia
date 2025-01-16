import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import librosa
import numpy as np
import soundfile as sf
import librosa.display
from bdd import url_collection,features_collection

def store_url():
    i=0
    url_collection.delete_many({})#vide la collection
    os.chdir("..")  # reculer d'un niveau dans la hierarchie
    os.chdir("sample")  # passer dans le dossier des musique
    repertoire = os.getcwd()# Pour avoir le repertoire dynamique pour le smusoiqur
    liste_musiques = os.listdir()
    liste_musiques = [repertoire + "\\" + element for element in liste_musiques]
    for elements in liste_musiques:
        url_collection.insert_one({'id' : i,'url': elements})
        i+=1

def extract_chromas(y,sr,fichier_audio):


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
    return chroma.tolist()
# Partie  MFCC

def extract_MFCC(y,sr):
    # Calcul des MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    #Normalisation
    mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
    return mfcc_normalized.tolist()

#Partie BPM
def extract_tempo(y,sr):

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = tempo[0]
    bpm = int(round(tempo))
    return bpm
def get_features(chemin_fichier):
    y, sr = librosa.load(chemin_fichier)
    y_trimmed, _ = librosa.effects.trim(y)
    chroma = extract_chromas(y_trimmed,sr,chemin_fichier)
    bpm = extract_tempo(y_trimmed,sr)
    mfcc=extract_MFCC(y_trimmed,sr)
    return chroma,bpm,mfcc


def analyse_musique():
    #Reinit la base d'url et la remplie
    store_url()
    #Récupérer tout les url de musiques
    cursor = url_collection
    for document in cursor.find({}, {"id": 1, "url": 1}):
        url = document['url']
        id  = document['id']
        chroma,bpm,mfcc = get_features(url)
        features_collection.insert_one({"id": id, "bpm":bpm,"chroma": chroma,"mfcc": mfcc })
        #print(url,chroma,bpm)
    #---------------------

analyse_musique()
