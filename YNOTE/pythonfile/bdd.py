from pymongo import MongoClient

client = MongoClient()

#Connexion a remplacer une fois la bdd hébergée
client = MongoClient("mongodb://localhost:27017/")

#db
database = client["database"]

# 1 collection pour les url, 1 pour les features
url_collection = database["url_collection"]
features_collection = database["features_collection"]