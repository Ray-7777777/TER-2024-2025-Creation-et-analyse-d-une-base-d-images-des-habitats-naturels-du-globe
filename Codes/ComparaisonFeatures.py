import torch
import pandas as pd
import numpy as np
import os
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 1. Chargement du modèle et création de l’extracteur de features
print("[DEBUG] Chargement du modèle ResNet50 avec extraction sur 'layer4.2.conv2'...")
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
base_model.eval()
return_nodes = {'layer4.2.conv2': 'feat_map'}
feature_extractor = create_feature_extractor(base_model, return_nodes)
print("[DEBUG] Modèle et extracteur initialisés.")

# 2. Transformation d'image sans resize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Fonction d’extraction de l’embedding (pooling avg ou max)
def extract_embedding(image_path, pooling='avg'):
    print(f"[DEBUG] Traitement de l'image : {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat_map = feature_extractor(t)['feat_map'].squeeze(0)  # (512, H, W)
        if pooling == 'avg':
            emb = feat_map.mean(dim=(1, 2))
        elif pooling == 'max':
            emb = feat_map.amax(dim=(1, 2))
        else:
            raise ValueError("pooling doit être 'avg' ou 'max'")
        print("[DEBUG] Extraction réussie.")
        return emb.cpu().numpy()
    except Exception as e:
        print(f"[ERREUR] Échec du traitement de l'image {image_path}: {e}")
        return None

# 4. Fonction principale de traitement par lot et sauvegarde CSV
def save_features_to_csv(folder_path, output_csv, batch_size=50, pooling='avg'):
    # Si le CSV existe déjà, on le supprime
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"[DEBUG] Ancien fichier {output_csv} supprimé.")

    print(f"[DEBUG] Dossier d'entrée : {folder_path}")
    if not os.path.exists(folder_path):
        print(f"[ERREUR] Le dossier spécifié n'existe pas: {folder_path}")
        return

    data = []
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        if '.ipynb_checkpoints' in root:
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    print(f"[DEBUG] {len(image_files)} images à traiter.")

    for idx, image_path in enumerate(image_files):
        embedding = extract_embedding(image_path, pooling)
        if embedding is not None:
            # on stocke [prefix, feat0, feat1, …]
            prefix = os.path.splitext(os.path.basename(image_path))[0]
            data.append([prefix] + embedding.tolist())

        # flush par batch
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(image_files):
            # on détermine dynamiquement le nombre de features
            n_feats = len(data[0]) - 1
            cols    = ['prefix'] + [f'f{i:03d}' for i in range(n_feats)]
            df      = pd.DataFrame(data, columns=cols)

            df.to_csv(output_csv, mode='a',
                      header=not os.path.exists(output_csv),
                      index=False)
            print(f"[DEBUG] {idx + 1}/{len(image_files)} images traitées et sauvegardées dans {output_csv}")
            data = []

    print("[DEBUG] Extraction et sauvegarde terminées.")

# 5. Exemple d'utilisation
image_folder = '../Donnees/birds_dataset/Background_extracted'
output_csv = '../Donnees/backgrounds_conv2_features.csv'


# Fonction pour moyenner les features par préfixe d'image
def mean_features_from_csv(input_csv, output_csv):
    # Charger les données du fichier CSV
    df = pd.read_csv(input_csv)
    
    # Vérifier que le CSV a les colonnes attendues
    if df.empty:
        print("[ERREUR] Le fichier CSV est vide ou mal formaté.")
        return
    
    # Extraire le préfixe en ne gardant que la partie après le dernier "\" et avant le dernier "_"
    df['prefix'] = df['0'].apply(lambda x: '_'.join(x.split('\\')[-1].split('_')[:-1]))
    
    # Conserver uniquement les colonnes numériques pour le calcul des moyennes
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    
    # Grouper les données par préfixe et calculer la moyenne des features numériques
    grouped = df_numeric.groupby(df['prefix']).agg(np.mean).reset_index()
    
    # ----- NOUVEAU : Renommer les colonnes numériques en f000, f001, … -----
    # on prend toutes les colonnes sauf 'prefix'
    old_feats = [c for c in grouped.columns if c != 'prefix']
    # on construit un mapping ancien → nouveau
    mapping = { old: f"f{idx:03d}" for idx, old in enumerate(old_feats) }
    # on renomme
    grouped = grouped.rename(columns=mapping)
    # -----------------------------------------------------------------------
    
    # Sauvegarder les résultats dans un nouveau fichier CSV
    grouped.to_csv(output_csv, index=False)
    print(f"[DEBUG] Features moyennes sauvegardées dans {output_csv}")

# Exemple d'utilisation
input_csv = '../Donnees/backgrounds_conv2_features.csv'  # Remplace par le chemin vers ton fichier CSV avec les features
output_csv = '../Donnees/backgrounds_ResNet_features_means.csv'  # Chemin pour sauvegarder les features moyennes

mean_features_from_csv(input_csv, output_csv)


# Charger les deux CSV
df_images = pd.read_csv('../Donnees/birds_conv2_features.csv')  # Features des images d'oiseaux
df_backgrounds = pd.read_csv('../Donnees/backgrounds_ResNet_features_means.csv')  # Features des backgrounds

# Renommer la colonne '0' dans df_images en 'prefix' pour faciliter la fusion
df_images.rename(columns={'0': 'prefix'}, inplace=True)

# Fusionner les deux DataFrames sur la colonne 'prefix' (nom de l'image)
df_merged = pd.merge(df_images, df_backgrounds, on='prefix', suffixes=('_image', '_background'))

if df_merged.empty:
    print("[ERREUR] Aucune correspondance entre les images et les backgrounds.")
else:
    print(f"[INFO] Nombre d'images avec un fond correspondant : {df_merged.shape[0]}")

# Extraire les features pour l'image et le fond
feature_cols_image      = [c for c in df_merged.columns if c.endswith('_image')]
feature_cols_background = [c for c in df_merged.columns if c.endswith('_background')]

image_features      = df_merged[feature_cols_image].values
background_features = df_merged[feature_cols_background].values

# Calculer la distance euclidienne pour chaque paire image/fond
distances = []
prefixes  = df_merged['prefix'].tolist()

for img_vec, bg_vec in zip(image_features, background_features):
    # Méthode 1 : avec pairwise_distances
    d = pairwise_distances(img_vec.reshape(1, -1),
                           bg_vec.reshape(1, -1),
                           metric='euclidean')[0, 0]
    # ou Méthode 2 : directement
    # d = np.linalg.norm(img_vec - bg_vec)
    distances.append(d)

# Créer le DataFrame des distances individuelles
distances_df = pd.DataFrame({
    'prefix': prefixes,
    'euclidean_distance': distances
})

# Calculer les mesures statistiques globales
mean_val     = distances_df['euclidean_distance'].mean()
median_val   = distances_df['euclidean_distance'].median()
q1_val       = np.percentile(distances_df['euclidean_distance'], 25)
q3_val       = np.percentile(distances_df['euclidean_distance'], 75)
variance_val = np.var(distances_df['euclidean_distance'])
std_val      = distances_df['euclidean_distance'].std()

# Créer un DataFrame pour les statistiques
stats_df = pd.DataFrame({
    'prefix': ['mean', 'median', 'q1', 'q3', 'variance', 'std'],
    'euclidean_distance': [mean_val, median_val, q1_val, q3_val, variance_val, std_val]
})

# Sauvegarder les résultats dans un même CSV
output_csv = '../Donnees/birds_background_euclidean_distance_results.csv'
distances_df.to_csv(output_csv, index=False)
stats_df.to_csv(output_csv, mode='a', index=False, header=False)

print("Le CSV a été sauvegardé avec les distances et les mesures statistiques.")
print("\nDistances individuelles :")
print(distances_df.head())


# 1) Charger et extraire l'espèce
df = pd.read_csv('../Donnees/backgrounds_ResNet_features_means.csv')
df['species'] = df['prefix'].str.rsplit('_', n=1).str[0]

# 2) Colonnes de features numériques
feature_cols = [c for c in df.columns if c not in ['prefix', 'species']]

# 3) Standardisation GLOBALE
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 4) Liste triée des espèces
species_list = sorted(df['species'].unique())

# 5) Boucle pour calculer toutes les stats par paire d'espèces
results = []
for sp1, sp2 in itertools.product(species_list, species_list):
    feats1 = df[df['species'] == sp1][feature_cols].values
    feats2 = df[df['species'] == sp2][feature_cols].values

    # intra‑espèce
    if sp1 == sp2:
        if len(feats1) < 2:
            distances = np.array([])
        else:
            dist_mat = pairwise_distances(feats1, metric='euclidean')
            idx = np.triu_indices_from(dist_mat, k=1)
            distances = dist_mat[idx]

    # inter‑espèce
    else:
        if feats1.size == 0 or feats2.size == 0:
            distances = np.array([])
        else:
            distances = pairwise_distances(feats1, feats2, metric='euclidean').ravel()

    results.append({
        'species_1':   sp1,
        'species_2':   sp2,
        'n_samples_1': len(feats1),
        'n_samples_2': len(feats2),
        'num_pairs':   len(distances),
        'mean_dist':   distances.mean()    if distances.size > 0 else np.nan,
        'median_dist': np.median(distances) if distances.size > 0 else np.nan,
        'q1_dist':     np.percentile(distances, 25) if distances.size > 0 else np.nan,
        'q3_dist':     np.percentile(distances, 75) if distances.size > 0 else np.nan,
        'var_dist':    distances.var()     if distances.size > 0 else np.nan,
        'std_dist':    distances.std()     if distances.size > 0 else np.nan,
    })

# 6) DataFrame et sauvegarde des statistiques détaillées
df_results = pd.DataFrame(results)
df_results.to_csv(
    '../Donnees/euclidean_distance_by_species_pair.csv',
    index=False
)

# 7) Pivot pour la matrice de confusion des moyennes
conf_matrix = df_results.pivot(
    index='species_1',
    columns='species_2',
    values='mean_dist'
)

# 8) Export en CSV de la matrice de confusion
conf_matrix.to_csv(
    '../Donnees/euclidean_mean_distance_confusion_matrix.csv',
    index=True
)
