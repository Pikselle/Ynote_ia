import numpy as np
from scipy.spatial.distance import euclidean

class Song:
    def __init__(self, song_id, chroma, bpm, mfcc, spectrogram):
        """
        Classe représentant une chanson avec ses features.

        :param song_id: Identifiant de la chanson
        :param chroma: Vecteur des caractéristiques chromatiques (array-like)
        :param bpm: Tempo de la chanson (float)
        :param mfcc: Vecteur MFCC (array-like)
        :param spectrogram: Vecteur du spectrogramme (array-like)
        """
        self.id = song_id
        self.chroma = np.array(chroma)
        self.bpm = float(bpm)
        self.mfcc = np.array(mfcc)
        self.spectrogram = np.array(spectrogram)

def get_features(file_path):
    """
    Fonction fictive d'extraction de features depuis un fichier MP3.
    À remplacer par une vraie implémentation.

    :param file_path: Chemin vers le fichier MP3
    :return: Dictionnaire avec les features
    """
    # Remplacer ceci par une vraie extraction audio
    return {
        'chroma': np.random.rand(12),
        'bpm': np.random.uniform(60, 180),
        'mfcc': np.random.rand(13),
        'spectrogram': np.random.rand(100)
    }

def compare_song(file_path, song_list, weights=None):
    """
    Compare les features d'un fichier MP3 à une liste d'objets Song et retourne le plus similaire.

    :param file_path: Chemin du fichier MP3 à comparer.
    :param song_list: Liste d'objets Song.
    :param weights: Dictionnaire de poids pour chaque feature (facultatif).
    :return: ID de la chanson la plus similaire.
    """
    # Poids par défaut
    default_weights = {'chroma': 1.0, 'bpm': 1.0, 'mfcc': 1.0, 'spectrogram': 1.0}
    if weights:
        default_weights.update(weights)

    try:
        target_features = get_features(file_path)
        target_song = Song("target", target_features['chroma'], target_features['bpm'],
                           target_features['mfcc'], target_features['spectrogram'])
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction des features : {e}")

    min_distance = float("inf")
    best_match = None

    for song in song_list:
        try:
            # Vérifie la compatibilité des dimensions
            dist_chroma = euclidean(target_song.chroma, song.chroma)
            dist_bpm = abs(target_song.bpm - song.bpm)
            dist_mfcc = euclidean(target_song.mfcc, song.mfcc)
            dist_spectrogram = euclidean(target_song.spectrogram, song.spectrogram)

            total_distance = (
                default_weights['chroma'] * dist_chroma +
                default_weights['bpm'] * dist_bpm +
                default_weights['mfcc'] * dist_mfcc +
                default_weights['spectrogram'] * dist_spectrogram
            )

            if total_distance < min_distance:
                min_distance = total_distance
                best_match = song
        except Exception as e:
            print(f"Erreur lors de la comparaison avec la chanson {song.id}: {e}")
            continue

    return best_match.id if best_match else None
