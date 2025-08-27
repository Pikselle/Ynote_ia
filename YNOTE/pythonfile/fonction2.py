import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import librosa
import numpy as np
import soundfile as sf
import librosa.display
from bdd import url_collection, features_collection
from sklearn.decomposition import PCA


# Fonction pour stocker les URL des musiques
def store_url():
    i = 0
    url_collection.delete_many({})  # Vide la collection
    os.chdir("..")  # Reculer d'un niveau dans la hiérarchie
    os.chdir("sample")  # Passer dans le dossier des musiques
    repertoire = os.getcwd()  # Pour avoir le répertoire dynamique pour les musiques
    liste_musiques = os.listdir()
    liste_musiques = [repertoire + "\\" + element for element in liste_musiques]
    for elements in liste_musiques:
        url_collection.insert_one({'id': i, 'url': elements})
        i += 1


# Partie Chroma
def extract_chromas(y, sr, fichier_audio):
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]

    X, sample_rate = sf.read(fichier_audio, dtype='float32')
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    return chroma.tolist()


def moyenne_voisins_groupes(arr, taille_groupe=3):

    i=1
    matrice_reduite = []
    if arr.size == 0:
        print("erreur dans la taille de l'array")
    for i in range(1, len(arr) - 1):
        if (i == 1 or (i - 1) % 5 == 0) and (i+5<=len(arr) - 1):
            new_val = (arr[i] + arr[i-1] + arr[i+1] + arr[i+2] + arr[i-2])/5
            matrice_reduite.append(new_val)
            print(i)

    return np.array(matrice_reduite)

# Partie MFCC avec réduction de taille
def extract_MFCC(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10, n_fft=512, hop_length=128)
    mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)

    # Réduction de la taille via reshape (exemple)
    mfcc_resized = moyenne_voisins_groupes(mfcc_normalized,5)

    # Retourner la version réduite en taille
    return mfcc_resized.tolist()


# Partie BPM
def extract_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = tempo[0]
    bpm = int(round(tempo))
    return bpm


# Partie spectrogramme avec réduction de taille
def extract_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=8, fmax=500, n_fft=256, hop_length=128)
    S_dB = librosa.power_to_db(S, ref=np.max)  # Conversion en échelle logarithmique (dB)

    # Réduction de la taille via reshape (exemple)
    S_dB_resized = S_dB.reshape(S_dB.shape[0], -1)  # Aplatir ou ajuster la forme

    # Appliquer PCA pour réduire la dimensionnalité
    pca = PCA(n_components=3)  # Réduction à 5 composantes principales
    S_dB_pca = pca.fit_transform(S_dB_resized.T)  # Appliquer PCA sur les frames temporelles

    print("Taille en byte", S_dB_pca.size * S_dB_pca.itemsize)
    print("Taille S_dB_pca", S_dB_pca.shape)

    return S_dB_pca.tolist(), S_dB.tolist()


# Fonction pour récupérer toutes les caractéristiques
def get_features(chemin_fichier):
    y, sr = librosa.load(chemin_fichier)
    y_trimmed, _ = librosa.effects.trim(y)
    chroma = extract_chromas(y_trimmed, sr, chemin_fichier)
    bpm = extract_tempo(y_trimmed, sr)
    mfcc = extract_MFCC(y_trimmed, sr)  # MFCC peut être une matrice 2D
    spectrogramme = extract_spectrogram(y_trimmed, sr)

    return chroma, bpm, spectrogramme, mfcc


# Fonction d'analyse et stockage dans MongoDB
def analyse_musique():
    # Reinit la base d'url et la remplie
    store_url()

    # Récupérer toutes les URLs des musiques
    cursor = url_collection
    for document in cursor.find({}, {"id": 1, "url": 1}):
        url = document['url']
        id = document['id']
        # Extraire les caractéristiques de chaque musiqu
        chroma, bpm, spectro, mfcc = get_features(url)
        print(f"valeur du spectrograme {spectro}" )
        print(f"valeur du mfcc {mfcc}" )
        # Insérer dans MongoDB
        features_collection.insert_one({
            "id": id,
            "bpm": bpm,
            "chroma": chroma,
            "spectrogramme": spectro,
            "mfcc": mfcc
        })
analyse_musique()

# Récupérer et afficher le premier document de la collection
first_document = features_collection.find_one()

# Afficher uniquement la première valeur (par exemple, le premier champ du premier document)
if first_document:
    print(first_document)