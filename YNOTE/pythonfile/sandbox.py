#fichier pour faire tout les test

##print(os.getcwd())  pour voir le dossier courant
#import os
#import shutil  #pour les test de répertoire
from bdd import url_collection, features_collection
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import librosa
import numpy as np
import soundfile as sf
import librosa.display

#from main import extract_chromas, extract_tempo
#---------------------------------------------
# Test de copie pour vérifier que les url fonctionen
#dst = "C:\\Users\\Theo\\PycharmProjects\\YNote\\Ynote_ia\\YNOTE\\pythonfile\\Bobby McFerrin - Dont Worry Be Happy (Official Music Video).mp3"
#shutil.copyfile(liste_musiques[0], dst)

#--------------------------------------------
#Verifier le contenu compelt d'une collection
#cursor = url_collection # choosing the collection you need
#for document in cursor.find():
#    print (document)

#file_car = url_collection.find_one({}, {"id": 1, "url": 1})
#chemin_fichier=file_car['url']

#Analyse en masse des musiques

cursor = features_collection  # choosing the collection you need
for document in cursor.find({}):
    print(document)


#chroma_test,bpm_test = analyse_musique()
#print(chroma_test)
#print(bpm_test)
