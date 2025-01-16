#fichier pour faire tout les test

##print(os.getcwd())  pour voir le dossier courant
import os

os.chdir("..") #reculer d'un niveau dans la hierarchie
os.chdir("sample")#passer dans le dossier des musique
repertoire = os.getcwd()
print("repertoire=",repertoire)
liste_musiques= os.listdir()
print(liste_musiques)


liste_musiques = ['%{0}%'.format(element) for element in liste_musiques]