from pymongo.mongo_client import MongoClient

# user admin mdp admin lol
uri = "mongodb+srv://admin:admin@clusterchords.5kcax.mongodb.net/?retryWrites=true&w=majority&appName=ClusterChords"

client = MongoClient(uri)

# Vérif de la connexion
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

database = client["database"]
url_collection = database["url_collection"]
features_collection = database["features_collection"]