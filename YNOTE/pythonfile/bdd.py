from pymongo import MongoClient

client = MongoClient()

#Connexion a remplacer une fois la bdd hébergée
client = MongoClient("mongodb://localhost:27017/")

#2 base, une pour les url(test uniquement les dev font la vrai) un pour les feature
url_db = client["url"]
feature_db = client["features"]


def get_url():
    return 0
