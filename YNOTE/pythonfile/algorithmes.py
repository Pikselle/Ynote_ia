from sklearn.neighbors import NearestNeighbors
from bdd import features_collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
def mise_en_forme_df():

    #Je recupere toute la collection avec .find()
    #par dessus je le transforme en liste avec list()
    #Et je transforme la liste en dataframe pour l'exploiter avec sklearn après
    # ensuite avec test_chroma je transforme les listes en colonne distinctes
    cursor = features_collection.find()
    cursor_list = cursor.to_list()
    df = pd.DataFrame(cursor_list)
    df = test_chroma(df)
    df.drop('_id',axis=1,inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 0)
    print(df.head())
    return df

def test_chroma(df):
    # Transformation de la colonne 'chroma' en plusieurs colonnes
    # Transformer les colonnes de listes en colonnes individuelles
    chroma_df = pd.DataFrame(df['chroma'].tolist()).add_prefix('chroma_')
    spectro_df = pd.DataFrame(df['spectro'].tolist()).add_prefix('spectro_')
    mfcc_df = pd.DataFrame(df['mfcc'].tolist()).add_prefix('mfcc_')
    # Fusionner avec les colonnes restantes
    df = pd.concat([df.drop(columns=['chroma', 'spectro', 'mfcc']), chroma_df, spectro_df, mfcc_df], axis=1)

    print(df.head())
    print(df.columns)
    return df

df = mise_en_forme_df()
# --- WRAPPER POUR GRIDSEARCH ---
class NNWrapper(BaseEstimator):
    def __init__(self, n_neighbors=2, leaf_size=30):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.model = None

    def fit(self, X, y=None):
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm='ball_tree',
                                      leaf_size=self.leaf_size)
        self.model.fit(X)
        return self

    def score(self, X, y=None):
        distances, _ = self.model.kneighbors(X)
        return -np.mean(distances)

# --- RECHERCHE D'HYPERPARAMÈTRES ---
param_grid = {
    'n_neighbors': [2, 3, 5, 10],
    'leaf_size': [10, 20, 30, 50]
}

grid_search = GridSearchCV(NNWrapper(), param_grid, cv=3)
grid_search.fit(df)

best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés :", best_params)

# --- ENTRAÎNEMENT DU MODÈLE FINAL ---
nearest_nb = NearestNeighbors(
    n_neighbors=best_params['n_neighbors'],
    algorithm='ball_tree',
    leaf_size=best_params['leaf_size']
).fit(df)

distances, indices = nearest_nb.kneighbors(df)

# --- PROJECTION 2D POUR VISUALISATION ---
pca = PCA(n_components=2)
df_2d = pca.fit_transform(df)

plt.figure(figsize=(10, 6))
plt.scatter(df_2d[:, 0], df_2d[:, 1], s=30, c='blue', label='Points')

# Relier chaque point à ses voisins
for i, neighbors in enumerate(indices):
    for j in neighbors:
        plt.plot([df_2d[i, 0], df_2d[j, 0]], [df_2d[i, 1], df_2d[j, 1]], 'k-', alpha=0.2)

plt.title("Plus proches voisins avec Ball Tree (PCA 2D)")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





