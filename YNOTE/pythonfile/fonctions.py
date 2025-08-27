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
from sklearn.decomposition import PCA

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

def extract_MFCC(y, sr):
    # Calcul des MFCC avec des paramètres minimisés
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)

    # Réduire à une seule colonne de MFCC pour un affichage simplifié (par exemple, juste la première frame)
    mfcc_reduit = mfcc[:, 0:3].tolist()  # Ne garder qu'une seule frame (ou les premières)

    return mfcc_reduit
#Partie BPM
def extract_tempo(y,sr):

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = tempo[0]
    bpm = int(round(tempo))
    return bpm

# Partie spectrogramme

def extract_spectrogram(y, sr):
    # Calcul du spectrogramme Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, fmax=2000, n_fft=256, hop_length=128)

    # Vérification du spectrogramme initial
    print("Spectrogramme avant PCA:", S.shape)

    # Appliquer PCA pour réduire la dimensionnalité (réduction à 20 composantes principales)
    pca = PCA(n_components=20)  # Réduction à 20 composantes principales
    S_pca = pca.fit_transform(S.T)  # Appliquer PCA sur les frames temporelles (transpose pour que chaque frame soit un échantillon)

    # Vérification après PCA
    print("Spectrogramme après PCA:", S_pca.shape)

    # Réduire à une seule valeur par composante en prenant la moyenne de chaque composante principale
    S_pca_reduit = np.mean(S_pca, axis=0).tolist()  # Moyenne de chaque composante principale

    return S_pca_reduit
def get_features(chemin_fichier):
    y, sr = librosa.load(chemin_fichier)
    y_trimmed, _ = librosa.effects.trim(y)
    chroma = extract_chromas(y_trimmed,sr,chemin_fichier)
    bpm = extract_tempo(y_trimmed,sr)
    mfcc=extract_MFCC(y_trimmed,sr)
    spectrogramme = extract_spectrogram(y_trimmed,sr)
    return chroma,bpm,spectrogramme,mfcc


def analyse_musique():
    #Reinit la base d'url et la remplie
    store_url()
    #Récupérer tout les url de musiques
    cursor = url_collection
    for document in cursor.find({}, {"id": 1, "url": 1}):
        url = document['url']
        id  = document['id']
        chroma,bpm,spectro,mfcc = get_features(url)
        data = {
            "chroma": chroma,
            "bpm": bpm,
            "spectro": spectro,
            "mfcc": mfcc
        }
        features_collection.insert_one(data)

analyse_musique()
cursor2 = features_collection  # choosing the collection you need
for document in cursor2.find():
    print(document)